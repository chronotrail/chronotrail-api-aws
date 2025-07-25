#!/bin/bash

# ChronoTrail API Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
REGION="us-east-1"
PROFILE=""
SKIP_BUILD=false
SKIP_DEPLOY=false

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENVIRONMENT    Target environment (dev, staging, prod) [default: dev]"
    echo "  -r, --region REGION             AWS region [default: us-east-1]"
    echo "  -p, --profile PROFILE           AWS profile to use"
    echo "  --skip-build                    Skip Docker image build"
    echo "  --skip-deploy                   Skip CDK deployment (build only)"
    echo "  -h, --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e dev                       Deploy to dev environment"
    echo "  $0 -e prod -p production        Deploy to prod with specific AWS profile"
    echo "  $0 --skip-build -e staging      Deploy to staging without rebuilding image"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
    exit 1
fi

# Set AWS profile if provided
if [[ -n "$PROFILE" ]]; then
    export AWS_PROFILE="$PROFILE"
    print_status "Using AWS profile: $PROFILE"
fi

# Set AWS region
export AWS_DEFAULT_REGION="$REGION"

print_status "Starting deployment to $ENVIRONMENT environment in $REGION region"

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v cdk &> /dev/null; then
        print_error "AWS CDK is not installed or not in PATH"
        print_status "Install with: npm install -g aws-cdk"
        exit 1
    fi
    
    print_status "All dependencies are available"
}

# Build Docker image
build_image() {
    if [[ "$SKIP_BUILD" == true ]]; then
        print_warning "Skipping Docker image build"
        return
    fi
    
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t chronotrail-api:latest .
    
    # Tag for ECR if not dev environment
    if [[ "$ENVIRONMENT" != "dev" ]]; then
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/chronotrail-api"
        
        # Create ECR repository if it doesn't exist
        aws ecr describe-repositories --repository-names chronotrail-api --region "$REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name chronotrail-api --region "$REGION"
        
        # Login to ECR
        aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"
        
        # Tag and push image
        docker tag chronotrail-api:latest "$ECR_URI:$ENVIRONMENT"
        docker tag chronotrail-api:latest "$ECR_URI:latest"
        docker push "$ECR_URI:$ENVIRONMENT"
        docker push "$ECR_URI:latest"
        
        print_status "Docker image pushed to ECR: $ECR_URI:$ENVIRONMENT"
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    if [[ "$SKIP_DEPLOY" == true ]]; then
        print_warning "Skipping CDK deployment"
        return
    fi
    
    print_status "Deploying infrastructure with CDK..."
    
    cd infrastructure/cdk
    
    # Install CDK dependencies
    pip install -r requirements.txt
    
    # Bootstrap CDK if needed
    cdk bootstrap --context environment="$ENVIRONMENT"
    
    # Deploy the stack
    cdk deploy --context environment="$ENVIRONMENT" --require-approval never
    
    cd ../..
    
    print_status "Infrastructure deployment completed"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Get database connection details from AWS Systems Manager
    DB_HOST=$(aws ssm get-parameter --name "/chronotrail/$ENVIRONMENT/database/host" --query 'Parameter.Value' --output text 2>/dev/null || echo "")
    
    if [[ -z "$DB_HOST" ]]; then
        print_warning "Database host not found in SSM. Skipping migrations."
        return
    fi
    
    # Run migrations using Docker
    docker run --rm \
        -e DATABASE_URL="postgresql://chronotrail:password@$DB_HOST:5432/chronotrail" \
        chronotrail-api:latest \
        alembic upgrade head
    
    print_status "Database migrations completed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Get the load balancer URL
    LB_URL=$(aws cloudformation describe-stacks \
        --stack-name "ChronoTrail-${ENVIRONMENT^}" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    if [[ -z "$LB_URL" ]]; then
        print_warning "Load balancer URL not found. Skipping health check."
        return
    fi
    
    # Wait for service to be healthy
    print_status "Waiting for service to be healthy at https://$LB_URL/health"
    
    for i in {1..30}; do
        if curl -f -s "https://$LB_URL/health" > /dev/null; then
            print_status "Service is healthy!"
            return
        fi
        print_status "Attempt $i/30: Service not ready yet, waiting 10 seconds..."
        sleep 10
    done
    
    print_error "Service health check failed after 5 minutes"
    exit 1
}

# Main deployment flow
main() {
    check_dependencies
    build_image
    deploy_infrastructure
    
    if [[ "$ENVIRONMENT" != "dev" ]]; then
        run_migrations
        health_check
    fi
    
    print_status "Deployment to $ENVIRONMENT completed successfully!"
    
    # Show useful information
    echo ""
    echo "=== Deployment Information ==="
    if [[ -n "$LB_URL" ]]; then
        echo "API URL: https://$LB_URL"
    fi
    echo "Environment: $ENVIRONMENT"
    echo "Region: $REGION"
    echo "AWS Profile: ${AWS_PROFILE:-default}"
    echo ""
    echo "Next steps:"
    echo "1. Configure your mobile app with the API URL"
    echo "2. Set up monitoring and alerting"
    echo "3. Configure your domain DNS if using custom domain"
}

# Run main function
main