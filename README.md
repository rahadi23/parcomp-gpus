# Parallel Computing with GPU

## Development

### Add a new task

To add a new task, create a task folder in `src` directory

```
mkdir -p mytask
```

Inside the folder, create a `main.cu` file. This will be the entry point of the task.

```
nano main.cu
```

### Add a task util

To include a specific utils for a task, create a `utils` folder inside the task's folder

```
mkdir -p mytask/utils
```

Inside the folder, create a utils file

```
nano helper.h
nano helper.c
```

And include the header file to the `main.cu` file

```C
extern "C"
{
#include "utils/helper.h"
}
```

### Global utils

There is also `src/utils` directory which is the global utils folder. The code inside this directory will be compiled for all tasks.

### Compiling the tasks

The tasks can be built using the `make` command

```
make
```

### Running the tasks

The compiled tasks will be available in `out` directory as `<taskname>.o`. To run `mytask.o`, use the following command:

```
./out/mytask.o
```

## Deployment

### DGX-1 Cluster

#### Preparation

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
or use `environments/dgx-1/pods.yml`

#### Deployment

Run the pod with the following command:

```
kubectl apply -f ./pods.yml
```

We assume that the kubectl's authentication is already set in the default config `~/.kube/config` to access the cluster. You can also set `--kubeconfig` flag to use a custom config file.


#### Monitoring

Observe the pod creation with the following command:

```
kubectl describe pods user05-nvhpc-cuda
```

The pod's name `user05-nvhpc-cuda` in this example should match the pod's name that is deployed via `pods.yml` earlier (see `metadata.name`).

What we want to monitor is the pod's events. We should make sure that the pod creation process is running as expected.

Furthermore, check the pod's status using the following command:

```
kubectl get pods
```

If everything went well, the pod's status should be 'running'.

#### Usage

We can use the pod by `exec`-ing to the pod:

```
kubectl exec -it user05-nvhpc-cuda -- /bin/bash
```

After that, we can do whatever we want inside the pod. For example:

##### Check GPU information

```
nvidia-smi
```

##### Compile & run a CUDA program

```
nvcc program.cu -o program.o
./program.o
```

#### Post Usage

We can do cleanup to pods that are no longer used by deleting it using the following command:

```
kubectl delete pods user05-nvhpc-cuda
```

