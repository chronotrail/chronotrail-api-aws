# ChronoTrail API Infrastructure

This directory contains the infrastructure as code (IaC) for deploying the ChronoTrail API to AWS using AWS CDK.

## Architecture Overview

The ChronoTrail API is deployed using the following AWS services:

- **ECS Fargate**: Container orchestration for the FastAPI application
- **Application Load Balancer**: Load balancing and SSL termination
- **RDS PostgreSQL**: Relational database for structured data
- **OpenSearch Serverless**: Vector database for semantic search
- **S3**: Object storage for media files
- **Cognito**: User authentication and management
- **Secrets Manager**: Secure storage of database credentials
- **Systems Manager Parameter Store**: Configuration management
- **Route53**: DNS management
- **Certificate Manager**: SSL/TLS certificates

## Prerequisites

### Required Tools

1. **AWS CLI** - Configure with appropriate credentials
   ```bash
   aws configure
   ```

2. **AWS CDK** - Install globally
   ```bash
   npm install -g aws-cdk
   ```

3. **Docker** - For building container images
   ```bash
   # Install Docker Desktop or Docker Engine
   ```

4. **Python 3.11+** - For CDK application
   ```bash
   python --version
   ```

### AWS Account Setup

1. **Bootstrap CDK** in your target regions:
   ```bash
   cdk bootstrap aws://ACCOUNT-NUMBER/REGION
   ```

2. **Create SSL Certificates** in AWS Certificate Manager for your domains:
   - `api.chronotrail.com` (production)
   - `staging-api.chronotrail.com` (staging)
   - `dev-api.chronotrail.com` (development)

3. **Set up Route53 Hosted Zone** for your domain

4. **Configure CDK Context** (see Configuration section below)

## Configuration

### CDK Context Configuration

Create a `cdk.context.json` file in the `infrastructure/cdk` directory:

```json
{
  "dev_account": "123456789012",
  "staging_account": "123456789012",
  "prod_account": "123456789012",
  "dev_certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/dev-cert-id",
  "staging_certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/staging-cert-id",
  "prod_certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/prod-cert-id"
}
```

### Environment Variables

Each environment uses different configuration files:

- **Development**: `config/dev.env`
- **Staging**: `config/staging.env`
- **Production**: `config/prod.env`

## Deployment

### Quick Deployment

Use the deployment script for automated deployment:

```bash
# Deploy to development
./scripts/deploy.sh -e dev

# Deploy to staging with specific AWS profile
./scripts/deploy.sh -e staging -p staging-profile

# Deploy to production
./scripts/deploy.sh -e prod -p production-profile
```

### Manual Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t chronotrail-api:latest .
   ```

2. **Deploy Infrastructure**:
   ```bash
   cd infrastructure/cdk
   pip install -r requirements.txt
   cdk deploy --context environment=dev
   ```

3. **Run Database Migrations**:
   ```bash
   # Get database endpoint from AWS
   DB_HOST=$(aws ssm get-parameter --name "/chronotrail/dev/database/host" --query 'Parameter.Value' --output text)
   
   # Run migrations
   docker run --rm \
     -e DATABASE_URL="postgresql://chronotrail:password@$DB_HOST:5432/chronotrail" \
     chronotrail-api:latest \
     alembic upgrade head
   ```

### Deployment Options

The deployment script supports several options:

- `--environment` / `-e`: Target environment (dev, staging, prod)
- `--region` / `-r`: AWS region (default: us-east-1)
- `--profile` / `-p`: AWS profile to use
- `--skip-build`: Skip Docker image build
- `--skip-deploy`: Skip CDK deployment (build only)

## Environment-Specific Configuration

### Development Environment

- **Purpose**: Local development and testing
- **Resources**: Minimal, cost-optimized
- **Features**: 
  - API documentation enabled
  - Debug logging
  - Relaxed security settings
  - Single AZ deployment

### Staging Environment

- **Purpose**: Pre-production testing
- **Resources**: Production-like but smaller scale
- **Features**:
  - Production-like configuration
  - Multi-AZ deployment
  - Performance monitoring
  - Automated testing integration

### Production Environment

- **Purpose**: Live production workloads
- **Resources**: High availability and performance
- **Features**:
  - Multi-AZ deployment
  - Auto-scaling enabled
  - Enhanced monitoring
  - Backup and disaster recovery
  - Security hardening

## Monitoring and Maintenance

### Health Checks

The application includes several health check endpoints:

- `/health` - Basic health status
- `/metrics` - Application metrics
- Load balancer health checks on port 8000

### Logging

Logs are centralized in CloudWatch:

- **Application logs**: `/ecs/chronotrail-{environment}`
- **Load balancer logs**: Stored in S3
- **Database logs**: CloudWatch Logs

### Monitoring

Set up CloudWatch alarms for:

- **Application**: Response time, error rate, CPU/memory usage
- **Database**: Connection count, CPU utilization, storage
- **Load Balancer**: Request count, target health
- **OpenSearch**: Cluster health, search latency

### Backup and Recovery

- **Database**: Automated backups with point-in-time recovery
- **Media Files**: S3 versioning and cross-region replication
- **Configuration**: Infrastructure as code in version control

## Security Considerations

### Network Security

- **VPC**: Isolated network with public/private subnets
- **Security Groups**: Restrictive ingress/egress rules
- **NACLs**: Additional network-level security
- **WAF**: Web Application Firewall (recommended for production)

### Data Security

- **Encryption at Rest**: All data encrypted (RDS, S3, OpenSearch)
- **Encryption in Transit**: TLS 1.2+ for all communications
- **Secrets Management**: AWS Secrets Manager for sensitive data
- **IAM**: Least privilege access principles

### Application Security

- **Authentication**: AWS Cognito with OAuth providers
- **Authorization**: JWT tokens with role-based access
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: API rate limiting and throttling

## Troubleshooting

### Common Issues

1. **CDK Bootstrap Error**:
   ```bash
   # Re-bootstrap the environment
   cdk bootstrap --force
   ```

2. **Docker Build Failures**:
   ```bash
   # Clear Docker cache
   docker system prune -a
   ```

3. **Database Connection Issues**:
   ```bash
   # Check security group rules
   aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
   ```

4. **OpenSearch Access Issues**:
   ```bash
   # Verify IAM permissions and network policies
   aws opensearchserverless get-security-policy --name chronotrail-dev-network --type network
   ```

### Useful Commands

```bash
# View stack outputs
aws cloudformation describe-stacks --stack-name ChronoTrail-Dev

# Check ECS service status
aws ecs describe-services --cluster ChronoTrail-Dev --services chronotrail-service

# View application logs
aws logs tail /ecs/chronotrail-dev --follow

# Connect to database
psql -h DB_HOST -U chronotrail -d chronotrail
```

## Cost Optimization

### Development Environment

- Use t3.micro instances where possible
- Single AZ deployment
- Minimal backup retention
- Scheduled shutdown for non-business hours

### Production Environment

- Use Reserved Instances for predictable workloads
- Enable S3 Intelligent Tiering
- Optimize OpenSearch instance types
- Regular cost reviews and rightsizing

## Support and Maintenance

### Regular Tasks

- **Weekly**: Review CloudWatch alarms and metrics
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and optimize costs
- **Annually**: Disaster recovery testing

### Scaling Considerations

The infrastructure is designed to scale automatically:

- **ECS Auto Scaling**: Based on CPU and memory utilization
- **Database**: Read replicas for read-heavy workloads
- **OpenSearch**: Automatic scaling based on usage
- **S3**: Unlimited storage capacity

For questions or issues, refer to the main project documentation or contact the development team.