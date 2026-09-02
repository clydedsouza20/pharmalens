# AWS Deployment

Deployed to ECS Fargate in `us-east-1`, verified on the live endpoint, torn
down. Notes reflect the actual run, including the two IAM gaps that turned up.

---

## Services used

| Service | Role | Cost |
|---|---|---|
| **ECR** | Private registry. Fargate can only pull from a registry, not from a laptop. | ~$0.14/month for 1.37GB |
| **ECS Fargate** | Runs the container. No servers to manage. | ~$0.05/hour |
| **CloudWatch Logs** | Container logs. Without the group, a failed task explains nothing. | Free at this volume |
| **IAM** | Execution role letting ECS pull from ECR and write logs. | Free |
| **VPC / Security Group** | Default VPC; port 8080 open. No EC2 instances. | Free |

## Deliberately not used

- **No S3.** The Chroma index is 2,857 documents. Baking it into the image
  makes the container stateless -- no bucket, no IAM binding, no startup
  fetch. S3 would add an external dependency for no benefit.
- **No load balancer.** An ALB is ~$16/month plus target groups and
  listeners. A public IP on the task does the same job for a demo. Not
  production practice; correct here.
- **No SageMaker.** Built for models needing GPUs, managed autoscaling, or
  A/B traffic splitting. The XGBoost model is 2MB and the embedding model
  runs on CPU.
- **No EKS.** The control plane alone is $73/month before a single node. One
  stateless container is exactly the Fargate use case.
- **No EC2 compute.** Fargate is EC2 with the host abstracted away. The task
  definition maps almost one-to-one onto a Kubernetes pod spec, which makes
  it the more transferable thing to learn.

---

## Steps

1. `aws ecr create-repository` with `scanOnPush=true`
2. `aws ecr get-login-password | docker login` -- 12-hour token
3. `docker tag` + `docker push` -- 1.37GB upload
4. `aws logs create-log-group /ecs/pharmalens` -- **before** any task runs
5. `aws iam create-role ecsTaskExecutionRole` + attach
   `AmazonECSTaskExecutionRolePolicy`
6. `aws ecs register-task-definition` -- see `taskdef.json`
7. `aws ecs create-cluster`
8. Security group opening TCP 8080 on the default VPC
9. `aws ecs run-task` with `assignPublicIp=ENABLED`
10. Read the public IP from the task's elastic network interface
11. Verify all four safety routes against the live endpoint
12. `aws ecs stop-task` + cleanup

---

## Three things that fail confusingly

**Log group before task.** If `/ecs/pharmalens` does not exist when a task
starts, the task dies immediately -- and because logging is what failed, there
are no logs saying why.

**`assignPublicIp=ENABLED`.** Without it the task cannot reach ECR to pull the
image. It fails with a timeout that never mentions networking.

**`startPeriod` long enough.** Health checks are ignored during the start
period. Set too short, ECS kills the task mid-boot and restarts it forever --
with logs that look healthy right up to the kill. Local startup was ~10s;
180s was used on Fargate's shared CPU.

---

## Task configuration

| Setting | Value | Reason |
|---|---|---|
| `cpu` | 1024 (1 vCPU) | Startup is CPU-bound building the resolver over 253,969 products |
| `memory` | 3072 (3GB) | Local peak ~1.2GB; headroom avoids an OOM kill that surfaces as a mystery crash |
| `startPeriod` | 180s | See above |
| `networkMode` | awsvpc | Required for Fargate; each task gets its own network interface |

---

## Measured

| | |
|---|---|
| Image size | 1.37GB (1,374,446,952 bytes) |
| Image pull | 53s |
| Container start | 2s |
| Total task startup | ~65s |
| Local container startup | 9.8s |
| REFUSE route latency, live | 27ms |
| Task runtime | ~35 min |
| Total cost | ~$0.03 |

The 53-second pull is the measured price of baking the model and index into
the image. That is the honest counterweight to "stateless is better": the
tradeoff was startup time for simplicity, and the number is known rather than
assumed.

---

## IAM gaps found

Two actions are missing from `AmazonECS_FullAccess` and
`AmazonEC2ContainerRegistryFullAccess`:

- `logs:PutRetentionPolicy` -- retention could not be set
- `ec2:DeleteSecurityGroup` -- the security group could not be deleted

Neither blocks deployment, and neither costs anything. `CloudWatchLogsFullAccess`
would cover the first; the second needs an EC2 policy. Both were left in place.

---

## Teardown

```bash
aws ecs stop-task --cluster pharmalens-cluster --task $TASK --region $REGION
aws ecs delete-cluster --cluster pharmalens-cluster --region $REGION
aws ecr delete-repository --repository-name pharmalens --force --region $REGION
```

The first command stops billing. The rest is housekeeping.

Fargate has no scale-to-zero -- a running task bills continuously at roughly
$30/month. A persistent demo would cost that for no additional signal, so the
task was torn down after verification.

---

## Verified on the live endpoint

```
what are the side effects of Crocin          route: ANSWER    docs: 1
side effects of flibanserin                  route: REFUSE    docs: 0
how many tablets of Dolo should I take       route: SCOPE     docs: 0
can I take Augmentin and Azithral together   route: PARTIAL   docs: 2
```

Screenshots in `reports/figures/`.
