"""
Main ChronoTrail infrastructure stack
"""

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_opensearchserverless as opensearch
from aws_cdk import aws_rds as rds
from aws_cdk import aws_route53 as route53
from aws_cdk import aws_route53_targets as targets
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_ssm as ssm
from constructs import Construct


class ChronoTrailStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        config: dict,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.environment = environment
        self.config = config

        # Create VPC
        self.vpc = self._create_vpc()

        # Create security groups
        self.security_groups = self._create_security_groups()

        # Create RDS database
        self.database = self._create_database()

        # Create OpenSearch domain
        self.opensearch_domain = self._create_opensearch()

        # Create S3 bucket for media files
        self.media_bucket = self._create_s3_bucket()

        # Create Cognito User Pool
        self.user_pool = self._create_cognito()

        # Create IAM roles
        self.task_role = self._create_task_role()

        # Create ECS cluster and service
        self.ecs_service = self._create_ecs_service()

        # Create Route53 records
        self._create_dns_records()

        # Create SSM parameters for configuration
        self._create_ssm_parameters()

        # Output important values
        self._create_outputs()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public and private subnets"""
        return ec2.Vpc(
            self,
            "ChronoTrailVPC",
            ip_addresses=ec2.IpAddresses.cidr(self.config["vpc_cidr"]),
            max_azs=3,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Database",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=28,
                ),
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True,
        )

    def _create_security_groups(self) -> dict:
        """Create security groups for different components"""
        # ALB Security Group
        alb_sg = ec2.SecurityGroup(
            self,
            "ALBSecurityGroup",
            vpc=self.vpc,
            description="Security group for Application Load Balancer",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS traffic")
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(80), "HTTP traffic (redirect to HTTPS)"
        )

        # ECS Security Group
        ecs_sg = ec2.SecurityGroup(
            self,
            "ECSSecurityGroup",
            vpc=self.vpc,
            description="Security group for ECS tasks",
            allow_all_outbound=True,
        )
        ecs_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8000), "Traffic from ALB")

        # RDS Security Group
        rds_sg = ec2.SecurityGroup(
            self,
            "RDSSecurityGroup",
            vpc=self.vpc,
            description="Security group for RDS database",
            allow_all_outbound=False,
        )
        rds_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(5432), "PostgreSQL from ECS")

        # OpenSearch Security Group
        opensearch_sg = ec2.SecurityGroup(
            self,
            "OpenSearchSecurityGroup",
            vpc=self.vpc,
            description="Security group for OpenSearch",
            allow_all_outbound=False,
        )
        opensearch_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(443), "HTTPS from ECS")

        return {
            "alb": alb_sg,
            "ecs": ecs_sg,
            "rds": rds_sg,
            "opensearch": opensearch_sg,
        }

    def _create_database(self) -> rds.DatabaseInstance:
        """Create RDS PostgreSQL database"""
        # Create database credentials secret
        db_credentials = rds.DatabaseSecret(
            self,
            "DBCredentials",
            username="chronotrail",
            description="ChronoTrail database credentials",
        )

        # Create subnet group
        subnet_group = rds.SubnetGroup(
            self,
            "DBSubnetGroup",
            description="Subnet group for ChronoTrail database",
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ),
        )

        # Create database instance
        database = rds.DatabaseInstance(
            self,
            "ChronoTrailDB",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_15_4
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MEDIUM
            ),
            credentials=rds.Credentials.from_secret(db_credentials),
            database_name="chronotrail",
            vpc=self.vpc,
            subnet_group=subnet_group,
            security_groups=[self.security_groups["rds"]],
            backup_retention=Duration.days(7 if self.environment == "prod" else 1),
            deletion_protection=self.config["enable_deletion_protection"],
            delete_automated_backups=not self.config["enable_deletion_protection"],
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.environment == "dev"
                else RemovalPolicy.SNAPSHOT
            ),
            allocated_storage=20,
            max_allocated_storage=100,
            storage_encrypted=True,
            monitoring_interval=Duration.seconds(60),
            enable_performance_insights=True,
            performance_insight_retention=rds.PerformanceInsightRetention.DEFAULT,
        )

        return database

    def _create_opensearch(self) -> opensearch.CfnCollection:
        """Create OpenSearch Serverless collection"""
        # Create encryption policy
        encryption_policy = opensearch.CfnSecurityPolicy(
            self,
            "OpenSearchEncryptionPolicy",
            name=f"chronotrail-{self.environment}-encryption",
            type="encryption",
            policy=f"""{{
                "Rules": [
                    {{
                        "ResourceType": "collection",
                        "Resource": ["collection/chronotrail-{self.environment}"]
                    }}
                ],
                "AWSOwnedKey": true
            }}""",
        )

        # Create network policy
        network_policy = opensearch.CfnSecurityPolicy(
            self,
            "OpenSearchNetworkPolicy",
            name=f"chronotrail-{self.environment}-network",
            type="network",
            policy=f"""[{{
                "Rules": [
                    {{
                        "ResourceType": "collection",
                        "Resource": ["collection/chronotrail-{self.environment}"]
                    }},
                    {{
                        "ResourceType": "dashboard",
                        "Resource": ["collection/chronotrail-{self.environment}"]
                    }}
                ],
                "AllowFromPublic": false,
                "SourceVPCEs": ["{self.vpc.vpc_id}"]
            }}]""",
        )

        # Create collection
        collection = opensearch.CfnCollection(
            self,
            "OpenSearchCollection",
            name=f"chronotrail-{self.environment}",
            type="VECTORSEARCH",
            description=f"ChronoTrail vector search collection for {self.environment}",
        )

        collection.add_dependency(encryption_policy)
        collection.add_dependency(network_policy)

        return collection

    def _create_s3_bucket(self) -> s3.Bucket:
        """Create S3 bucket for media file storage"""
        bucket = s3.Bucket(
            self,
            "MediaBucket",
            bucket_name=f"chronotrail-{self.environment}-media-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.environment == "dev"
                else RemovalPolicy.RETAIN
            ),
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteIncompleteMultipartUploads",
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
                s3.LifecycleRule(
                    id="TransitionToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        ),
                    ],
                ),
            ],
        )

        # Add CORS configuration for web uploads
        bucket.add_cors_rule(
            allowed_methods=[
                s3.HttpMethods.GET,
                s3.HttpMethods.POST,
                s3.HttpMethods.PUT,
            ],
            allowed_origins=["*"],  # Configure this based on your frontend domains
            allowed_headers=["*"],
            max_age=3000,
        )

        return bucket

    def _create_cognito(self) -> cognito.UserPool:
        """Create Cognito User Pool for authentication"""
        user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=f"chronotrail-{self.environment}",
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
                given_name=cognito.StandardAttribute(required=False, mutable=True),
                family_name=cognito.StandardAttribute(required=False, mutable=True),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.environment == "dev"
                else RemovalPolicy.RETAIN
            ),
        )

        # Create user pool client
        user_pool_client = cognito.UserPoolClient(
            self,
            "UserPoolClient",
            user_pool=user_pool,
            user_pool_client_name=f"chronotrail-{self.environment}-client",
            generate_secret=False,  # For mobile apps
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                admin_user_password=True,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    authorization_code_grant=True,
                    implicit_code_grant=True,
                ),
                scopes=[
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[
                    f"https://{self.config['domain_name']}/auth/callback",
                    "http://localhost:3000/auth/callback",  # For development
                ],
            ),
            supported_identity_providers=[
                cognito.UserPoolClientIdentityProvider.COGNITO,
                cognito.UserPoolClientIdentityProvider.GOOGLE,
                cognito.UserPoolClientIdentityProvider.APPLE,
            ],
        )

        return user_pool

    def _create_task_role(self) -> iam.Role:
        """Create IAM role for ECS tasks"""
        task_role = iam.Role(
            self,
            "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Role for ChronoTrail ECS tasks",
        )

        # Add policies for AWS services
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                ],
                resources=[f"{self.media_bucket.bucket_arn}/*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:ListBucket",
                ],
                resources=[self.media_bucket.bucket_arn],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "aoss:APIAccessAll",
                ],
                resources=[self.opensearch_domain.attr_arn],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "textract:DetectDocumentText",
                    "textract:AnalyzeDocument",
                ],
                resources=["*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "transcribe:StartTranscriptionJob",
                    "transcribe:GetTranscriptionJob",
                ],
                resources=["*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "rekognition:DetectText",
                    "rekognition:DetectLabels",
                ],
                resources=["*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "cognito-idp:GetUser",
                    "cognito-idp:AdminGetUser",
                ],
                resources=[self.user_pool.user_pool_arn],
            )
        )

        return task_role

    def _create_ecs_service(self) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        """Create ECS Fargate service with Application Load Balancer"""
        # Create log group
        log_group = logs.LogGroup(
            self,
            "ECSLogGroup",
            log_group_name=f"/ecs/chronotrail-{self.environment}",
            retention=(
                logs.RetentionDays.ONE_WEEK
                if self.environment == "dev"
                else logs.RetentionDays.ONE_MONTH
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Get certificate
        certificate = (
            acm.Certificate.from_certificate_arn(
                self, "Certificate", self.config["certificate_arn"]
            )
            if self.config.get("certificate_arn")
            else None
        )

        # Create Fargate service
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ChronoTrailService",
            vpc=self.vpc,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=self.config["desired_capacity"],
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset("."),
                container_port=8000,
                task_role=self.task_role,
                environment={
                    "ENVIRONMENT": self.environment,
                    "AWS_DEFAULT_REGION": self.region,
                },
                secrets={
                    "DATABASE_URL": ecs.Secret.from_secrets_manager(
                        self.database.secret, "engine"
                    ),
                    "OPENSEARCH_URL": ecs.Secret.from_ssm_parameter(
                        ssm.StringParameter.from_string_parameter_name(
                            self,
                            "OpenSearchURL",
                            f"/chronotrail/{self.environment}/opensearch/url",
                        )
                    ),
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="ecs",
                    log_group=log_group,
                ),
            ),
            domain_name=self.config["domain_name"] if certificate else None,
            domain_zone=(
                route53.HostedZone.from_lookup(
                    self, "HostedZone", domain_name="chronotrail.com"
                )
                if certificate
                else None
            ),
            certificate=certificate,
            redirect_http=True if certificate else False,
            public_load_balancer=True,
        )

        # Configure health check
        service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Configure auto scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=self.config["min_capacity"],
            max_capacity=self.config["max_capacity"],
        )

        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(2),
        )

        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(2),
        )

        return service

    def _create_dns_records(self):
        """Create Route53 DNS records"""
        if not self.config.get("certificate_arn"):
            return

        hosted_zone = route53.HostedZone.from_lookup(
            self, "HostedZone", domain_name="chronotrail.com"
        )

        # A record for the API
        route53.ARecord(
            self,
            "APIRecord",
            zone=hosted_zone,
            record_name=self.config["domain_name"],
            target=route53.RecordTarget.from_alias(
                targets.LoadBalancerTarget(self.ecs_service.load_balancer)
            ),
        )

    def _create_ssm_parameters(self):
        """Create SSM parameters for application configuration"""
        # OpenSearch endpoint
        ssm.StringParameter(
            self,
            "OpenSearchEndpoint",
            parameter_name=f"/chronotrail/{self.environment}/opensearch/url",
            string_value=f"https://{self.opensearch_domain.attr_collection_endpoint}",
            description="OpenSearch collection endpoint",
        )

        # S3 bucket name
        ssm.StringParameter(
            self,
            "MediaBucketName",
            parameter_name=f"/chronotrail/{self.environment}/s3/media-bucket",
            string_value=self.media_bucket.bucket_name,
            description="S3 bucket for media files",
        )

        # Cognito User Pool ID
        ssm.StringParameter(
            self,
            "CognitoUserPoolId",
            parameter_name=f"/chronotrail/{self.environment}/cognito/user-pool-id",
            string_value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID",
        )

    def _create_outputs(self):
        """Create CloudFormation outputs"""
        cdk.CfnOutput(
            self,
            "LoadBalancerDNS",
            value=self.ecs_service.load_balancer.load_balancer_dns_name,
            description="Load Balancer DNS name",
        )

        cdk.CfnOutput(
            self,
            "DatabaseEndpoint",
            value=self.database.instance_endpoint.hostname,
            description="RDS database endpoint",
        )

        cdk.CfnOutput(
            self,
            "OpenSearchEndpoint",
            value=self.opensearch_domain.attr_collection_endpoint,
            description="OpenSearch collection endpoint",
        )

        cdk.CfnOutput(
            self,
            "MediaBucketName",
            value=self.media_bucket.bucket_name,
            description="S3 media bucket name",
        )

        cdk.CfnOutput(
            self,
            "UserPoolId",
            value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID",
        )
