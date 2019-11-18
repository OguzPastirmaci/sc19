# Demo: GPU enabled Kubernetes cluster running on Oracle Cloud Infrastructure and Microsoft Azure using the interconnect

## 1. Running `nvidia-smi` on Kubernetes worker nodes

Let's start with running a very simple pod that shows the output of the `nvidia-smi` command in our Kubernetes cluster. 

### Running `nvidia-smi` on OCI

The worker nodes in Kubernetes are already labeled depending on the cloud they are running (worker in OCI has the `cloud: oci` label, and worker in Azure has the `cloud: azure` label. As you can see from the command, we will be selecting the nodes using this labels. You can also run the same with a yaml file.

```console
kubectl run oci-gpu-test --namespace supercomputing19 --rm -t -i --restart=Never --image=nvidia/cuda:10.1-base --limits=nvidia.com/gpu=1 --overrides='{"apiVersion": "v1", "spec": {"nodeSelector": { "cloud": "oci" }}}' -- nvidia-smi
```
You should see an output like below. Note that the VM on OCI has a **Nvidia Tesla V100** GPU.

```console
Mon Nov 18 19:58:37 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   34C    P0    20W / 300W |      0MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Running `nvidia-smi` on Azure

Now let's run the same command in Azure. This time we will select the node that has `cloud: azure` as the label:
```console
kubectl run azure-gpu-test --namespace supercomputing19 --rm -t -i --restart=Never --image=nvidia/cuda:10.1-base --limits=nvidia.com/gpu=1 --overrides='{"apiVersion": "v1", "spec": {"nodeSelector": { "cloud": "azure" }}}' -- nvidia-smi
```

You should see an output like below. Note that the VM on Azure has a **Nvidia Tesla K80** GPU.

```console
Mon Nov 18 20:02:28 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 0000D43D:00:00.0 Off |                    0 |
| N/A   39C    P8    33W / 149W |      0MiB / 11441MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## 2. Running a Tensorflow job
In this step, we will run a simple MNIST classifier which displays summaries as a Tensorflow job. The job will run on a single node, so it is not distributed.

Here's the yaml we will be running:

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: tf-mnist
spec:
  CleanPodPolicy: All
  tfReplicaSpecs:
    MASTER:
      replicas: 1
      template:
        spec:
          containers:
            - image: oguzpastirmaci/tf-mnist:gpu
              name: tensorflow
              resources:
                limits:
                  nvidia.com/gpu: 1
          restartPolicy: OnFailure
```

1. Let's run the Tensorflow job with the following command:
```console
kubectl create -f https://raw.githubusercontent.com/OguzPastirmaci/sc19/master/examples/tf-mnist.yaml
```

**NOTE:** If you receive an error message saying `Error from server (AlreadyExists): error when creating "https://raw.githubusercontent.com/OguzPastirmaci/sc19/master/examples/tf-mnist.yaml": tfjobs.kubeflow.org "tf-mnist" already exists`, run the following command to delete the existing job, and rerun the previous command again:

```console
kubectl delete -f https://raw.githubusercontent.com/OguzPastirmaci/sc19/master/examples/tf-mnist.yaml
```

2. Now run the following command to get the logs from the job:

```console
kubectl logs tf-mnist-master-0 -f
```

You should be seeing the logs. It will take about a minute for the job to complete.

```console
...
Adding run metadata for 799
Accuracy at step 800: 0.965
Accuracy at step 810: 0.9664
Accuracy at step 820: 0.966
Accuracy at step 830: 0.9672
Accuracy at step 840: 0.9664
Accuracy at step 850: 0.9667
Accuracy at step 860: 0.9647
Accuracy at step 870: 0.9633
Accuracy at step 880: 0.9669
Accuracy at step 890: 0.9646
Adding run metadata for 899
Accuracy at step 900: 0.9655
Accuracy at step 910: 0.965
Accuracy at step 920: 0.9662
Accuracy at step 930: 0.9689
Accuracy at step 940: 0.9684
Accuracy at step 950: 0.9676
Accuracy at step 960: 0.9679
Accuracy at step 970: 0.968
Accuracy at step 980: 0.9683
Accuracy at step 990: 0.969
Adding run metadata for 999


