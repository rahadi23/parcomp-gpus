# Parallel Computing with GPU

## Parallel Computing on DGX-1 Cluster

### Preparation

Create a pod file, for example `pods.yml`

```
nano pods.yml
```

with the following configuration:

```yml
apiVersion: v1
kind: Pod
metadata:
  # Pod Name
  name: user05-nvhpc-cuda
spec:
  restartPolicy: Never
  volumes:
      # Volume Name, should match spec.containers.volumeMounts.name
    - name: user05-nvhpc-cuda-pv
      persistentVolumeClaim:
        claimName: pemrograman-paralel-pv-claim
  containers:
    - name: app
      # Container Image
      image: "nvcr.io/nvidia/nvhpc:24.9-devel-cuda_multi-ubuntu22.04"
      command: ["/bin/sh"]
      args: ["-c", "while true; do echo 'pemrograman-paralel-pod log'; sleep 10; done"]
      volumeMounts:
          # Volume Name, should match spec.volumes.name
        - name: user05-nvhpc-cuda-pv
          mountPath: "/workspace"
```

### Deployment

Run the pod with the following command:

```
kubectl --kubeconfig ./dgx-config apply -f ./pods.yml
```

We set `--kubeconfig` flag to use a custom config file `./dgx-config` instead of the default config file `~/.kube/config`.
Remove this flag if you want to use the default config file.


### Monitoring

Observe pod creation with the following command:

```
kubectl --kubeconfig ./dgx-config describe pods user05-nvhpc-cuda
```

Again, we use custom config here. You can adjust these accordingly, including the pod's name.

What we want to monitor is the pod's events. We should make sure that the pod's creation process is running as expected.

Furthermore, check the pod's status using the following command:

```
kubectl --kubeconfig ./dgx-config get pods
```

If everything went well, the pod's status should be 'running'.

### Usage

We can use the pod by `exec`-ing to the pod:

```
kubectl --kubeconfig ./dgx-config exec -it user05-nvhpc-cuda /bin/bash
```

After that, we can do whatever we want inside the pod. For example:

#### Check GPU information

```
nvidia-smi
```

#### Compile & run a CUDA program

```
nvcc program.cu -o program.o
./program.o
```

### Post Usage

We can do cleanup to pods that are no longer used by deleting it using the following command:

```
kubectl --kubeconfig ./dgx-config delete pods user05-nvhpc-cuda
```

