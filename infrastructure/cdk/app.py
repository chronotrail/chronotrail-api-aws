#!/usr/bin/env python3
"""
AWS CDK app for ChronoTrail API infrastructure
"""

import aws_cdk as cdk
from constructs import Construct
from stacks.chronotrail_stack import ChronoTrailStack


class ChronoTrailApp(cdk.App):
    def __init__(self):
        super().__init__()

        # Get environment configuration
        env_name = self.node.try_get_context("environment") or "dev"

        # Environment-specific configuration
        env_config = {
            "dev": {
                "account": self.node.try_get_context("dev_account"),
                "region": "us-east-1",
                "domain_name": "dev-api.chronotrail.com",
                "certificate_arn": self.node.try_get_context("dev_certificate_arn"),
                "vpc_cidr": "10.0.0.0/16",
                "enable_deletion_protection": False,
                "min_capacity": 1,
                "max_capacity": 3,
                "desired_capacity": 1,
                "instance_class": "t3.medium",
                "opensearch_instance_type": "t3.small.search",
                "opensearch_instance_count": 1,
            },
            "staging": {
                "account": self.node.try_get_context("staging_account"),
                "region": "us-east-1",
                "domain_name": "staging-api.chronotrail.com",
                "certificate_arn": self.node.try_get_context("staging_certificate_arn"),
                "vpc_cidr": "10.1.0.0/16",
                "enable_deletion_protection": True,
                "min_capacity": 2,
                "max_capacity": 10,
                "desired_capacity": 2,
                "instance_class": "t3.large",
                "opensearch_instance_type": "t3.medium.search",
                "opensearch_instance_count": 2,
            },
            "prod": {
                "account": self.node.try_get_context("prod_account"),
                "region": "us-east-1",
                "domain_name": "api.chronotrail.com",
                "certificate_arn": self.node.try_get_context("prod_certificate_arn"),
                "vpc_cidr": "10.2.0.0/16",
                "enable_deletion_protection": True,
                "min_capacity": 3,
                "max_capacity": 20,
                "desired_capacity": 3,
                "instance_class": "c5.xlarge",
                "opensearch_instance_type": "r6g.large.search",
                "opensearch_instance_count": 3,
            },
        }

        config = env_config.get(env_name, env_config["dev"])

        # Create the main stack
        ChronoTrailStack(
            self,
            f"ChronoTrail-{env_name.title()}",
            env=cdk.Environment(account=config["account"], region=config["region"]),
            environment=env_name,
            config=config,
            description=f"ChronoTrail API infrastructure for {env_name} environment",
        )


if __name__ == "__main__":
    app = ChronoTrailApp()
    app.synth()
