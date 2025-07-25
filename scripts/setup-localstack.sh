#!/bin/bash

# Setup script for LocalStack AWS resources
set -e

echo "🚀 Setting up LocalStack AWS resources..."

# Wait for LocalStack to be ready
echo "⏳ Waiting for LocalStack to start..."
until curl -s http://localhost:4566/_localstack/health | grep -q '"s3": "available"'; do
    echo "Waiting for LocalStack..."
    sleep 2
done

echo "✅ LocalStack is ready!"

# Set LocalStack endpoint
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# Create S3 bucket
echo "📦 Creating S3 bucket..."
aws --endpoint-url=http://localhost:4566 s3 mb s3://chronotrail-media
aws --endpoint-url=http://localhost:4566 s3api put-bucket-cors --bucket chronotrail-media --cors-configuration '{
    "CORSRules": [
        {
            "AllowedOrigins": ["*"],
            "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
            "AllowedHeaders": ["*"],
            "MaxAgeSeconds": 3000
        }
    ]
}'

# Create Cognito User Pool
echo "👤 Creating Cognito User Pool..."
USER_POOL_ID=$(aws --endpoint-url=http://localhost:4566 cognito-idp create-user-pool \
    --pool-name chronotrail-users \
    --policies '{
        "PasswordPolicy": {
            "MinimumLength": 8,
            "RequireUppercase": false,
            "RequireLowercase": false,
            "RequireNumbers": false,
            "RequireSymbols": false
        }
    }' \
    --query 'UserPool.Id' --output text)

echo "User Pool ID: $USER_POOL_ID"

# Create Cognito User Pool Client
CLIENT_ID=$(aws --endpoint-url=http://localhost:4566 cognito-idp create-user-pool-client \
    --user-pool-id $USER_POOL_ID \
    --client-name chronotrail-client \
    --generate-secret \
    --explicit-auth-flows ADMIN_NO_SRP_AUTH USER_PASSWORD_AUTH \
    --query 'UserPoolClient.ClientId' --output text)

echo "Client ID: $CLIENT_ID"

# Create a test user
echo "👤 Creating test user..."
aws --endpoint-url=http://localhost:4566 cognito-idp admin-create-user \
    --user-pool-id $USER_POOL_ID \
    --username testuser \
    --user-attributes Name=email,Value=test@example.com \
    --temporary-password TempPass123! \
    --message-action SUPPRESS

# Set permanent password
aws --endpoint-url=http://localhost:4566 cognito-idp admin-set-user-password \
    --user-pool-id $USER_POOL_ID \
    --username testuser \
    --password TestPass123! \
    --permanent

echo "✅ LocalStack setup complete!"
echo ""
echo "📋 Configuration Summary:"
echo "  S3 Bucket: chronotrail-media"
echo "  User Pool ID: $USER_POOL_ID"
echo "  Client ID: $CLIENT_ID"
echo "  Test User: testuser / TestPass123!"
echo ""
echo "🔧 Add these to your .env file:"
echo "COGNITO_USER_POOL_ID=$USER_POOL_ID"
echo "COGNITO_CLIENT_ID=$CLIENT_ID"
echo "AWS_ENDPOINT_URL=http://localhost:4566"