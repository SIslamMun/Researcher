# Analysis of the Jarvis Framework for HPC Deployment and Container Orchestration

## Executive Summary

The Jarvis framework (specifically **Jarvis-CD**) is a specialized platform designed to address the complexity of deploying high-performance computing (HPC) applications, particularly those involving intricate storage configurations and benchmarking pipelines. Unlike general-purpose container orchestrators (e.g., Kubernetes) that dominate cloud environments, Jarvis focuses on the unique requirements of HPC hardware, such as MPI-based parallelism, shared parallel file systems, and hardware-specific resource mapping.

The framework operates on the concept of **"pkgs"** (packages) and **"pipelines."** It abstracts applications—whether they are long-running services like storage systems, transient benchmarks, or interception libraries—into modular components that can be composed into reproducible deployment flows. Its architecture relies on a **Resource Graph** to map software requirements to physical hardware topology, ensuring that data-intensive applications are aware of network and storage locality. While primarily documented as a bare-metal deployment tool integrated with **Spack** and **MPI**, it is positioned within broader ecosystems (such as the **IOWarp** and **LABIOS** projects) that support containerized workflows, bridging the gap between traditional HPC job scheduling and modern, flexible service orchestration.

---

## 1. Introduction

Deploying software in High-Performance Computing (HPC) environments presents distinct challenges compared to standard enterprise IT. Applications often require precise control over process placement, access to specialized networks (e.g., InfiniBand), and configuration of distributed storage systems (e.g., OrangeFS, BeeGFS). The **Jarvis** framework, comprising **Jarvis-CD** (Continuous Delivery) and its utility library **jarvis-util**, was developed to unify and automate these tasks [cite: 1].

Jarvis is defined as a "unified platform for deploying various applications, including storage systems and benchmarks" [cite: 1]. Its primary motivation is to manage the "complex configuration spaces" of storage systems and scientific workflows that are difficult to replicate across different machines [cite: 1]. By treating applications as modular "pkgs" that can be linked into "deployment pipelines," Jarvis provides a structured approach to defining, executing, and reproducing complex HPC experiments [cite: 1].

---

## 2. Architecture

The architecture of Jarvis is centered around the separation of application logic from deployment configuration, enabling portability across diverse HPC clusters. The system is built upon two main software components: **Jarvis-CD**, which handles the orchestration logic, and **jarvis-util**, which provides the low-level execution primitives [cite: 1].

### 2.1 The Pipeline Model
The core architectural abstraction in Jarvis is the **Pipeline**. A pipeline represents a complete, reproducible workflow consisting of an ordered set of configured packages ("pkgs") [cite: 2]. This model allows users to define a sequence of operations—such as starting a storage service, launching a benchmark, and then tearing down the service—as a single atomic unit.

Pipelines are stateful entities. When a pipeline is created, Jarvis generates metadata within a configured `CONFIG_DIR` [cite: 2]. This metadata persists the specific configuration parameters for every package in the pipeline, ensuring that the deployment can be repeated or modified without starting from scratch. This contrasts with stateless job scripts often used in HPC, where configuration is ephemeral and lost after execution.

### 2.2 The Resource Graph
To handle the heterogeneity of HPC hardware, Jarvis employs a **Resource Graph** [cite: 1, 2]. This is a YAML-based representation of the machine's state, capturing critical details about:
*   **Hardware:** CPU cores, memory, and storage devices (NVMe, SSD, HDD).
*   **Networks:** Available interfaces, IP addresses, and fabric types (e.g., TCP/IP, InfiniBand) [cite: 2].

The Resource Graph allows packages to query the environment dynamically. For example, a storage system package like Hermes can consult the Resource Graph to identify valid high-speed networks for data transfer or locate specific buffering devices (e.g., NVMe drives) to use for caching [cite: 1]. This decoupling of hardware specifics from application configuration is a key architectural feature that distinguishes Jarvis from static deployment scripts.

### 2.3 Configuration Management
Jarvis manages state through a hierarchy of directories defined in its global configuration (`jarvis_config.yaml`) [cite: 2, 3]:

*   **`CONFIG_DIR`:** Stores metadata for pipelines and packages. This is the control plane data, accessible to the user [cite: 3].
*   **`SHARED_DIR`:** A directory visible to all nodes in the cluster (e.g., on a parallel file system like Lustre or NFS). This is used for shared state, such as configuration files that must be read by all MPI ranks [cite: 3].
*   **`PRIVATE_DIR`:** A directory that exists locally on each node (e.g., `/tmp` or local scratch). This is used for node-specific data, such as local storage logs or process IDs [cite: 3].

This three-tier storage model addresses the specific needs of distributed systems where some data must be globally consistent while other data must remain local to a compute node.

---

## 3. Key Components

The Jarvis framework categorizes software into three distinct types of "pkgs," each serving a specific role in the orchestration lifecycle [cite: 2].

### 3.1 Service Packages
**Service Pkgs** represent long-running applications that operate as daemons or background processes [cite: 2]. In the context of HPC, these are typically distributed storage systems or data services.
*   **Behavior:** A service runs indefinitely until forcibly stopped.
*   **Complexity:** These packages handle the most complex configurations. For a storage system, the Service Pkg must determine which nodes act as metadata servers, which act as storage servers, what block devices to utilize, and what networking protocols to bind to [cite: 2].
*   **Examples:** OrangeFS, BeeGFS, and Hermes [cite: 2].

### 3.2 Application Packages
**Application Pkgs** represent finite tasks that run to completion [cite: 2]. These are typically the scientific workloads or benchmarks that utilize the underlying services.
*   **Behavior:** Runs to a definite completion point.
*   **Examples:** Standard HPC benchmarks like **IOR** (Interleaved or Random) or simulation codes like **Gray Scott** [cite: 2, 4].
*   **Integration:** An application package is often the "consumer" in a pipeline, configured to direct its I/O operations toward the mount points or APIs exposed by preceding Service Pkgs.

### 3.3 Interceptor Packages
**Interceptor Pkgs** are a specialized component designed to modify or monitor the behavior of other applications transparently [cite: 2].
*   **Mechanism:** These packages typically employ techniques like `LD_PRELOAD` to inject shared libraries into the address space of an Application Pkg.
*   **Use Cases:**
    *   **Profiling:** Capturing I/O calls to analyze performance (e.g., counting `read`/`write` operations).
    *   **Redirection:** Transparently redirecting POSIX I/O calls to a specialized storage backend (e.g., redirecting standard file operations to a burst buffer or object store) [cite: 2].
*   **Orchestration:** In a Jarvis pipeline, an Interceptor is "appended" to an Application, ensuring the environment variables (like `LD_PRELOAD`) are set correctly before the application binary executes.

---

## 4. Integration with Runtime Environments

Jarvis is designed to integrate deeply with the standard HPC software stack, rather than replacing it. It acts as a middleware layer that coordinates existing tools.

### 4.1 MPI and Hostfile Management
High-Performance Computing relies heavily on the Message Passing Interface (MPI) for distributed execution. Jarvis integrates with this paradigm by managing **Hostfiles** [cite: 3].
*   **Structure:** Jarvis uses hostfiles formatted similarly to traditional MPI hostfiles, listing the nodes available for a pipeline [cite: 3].
*   **Orchestration:** When a pipeline is executed, Jarvis uses this hostfile to distribute tasks. For example, if a storage service needs to run on 4 nodes and a benchmark on 8 nodes, Jarvis calculates the appropriate rank distribution and generates the necessary `mpirun` or `srun` commands via `jarvis-util` [cite: 1].

### 4.2 Spack Integration
For package management and installation, Jarvis leverages **Spack**, the de facto standard package manager for HPC [cite: 1].
*   **Dependency Management:** Jarvis itself is installable via Spack (`spack install py-ppi-jarvis-cd`) [cite: 1].
*   **Environment Loading:** It relies on Spack's module system to load dependencies. The documentation notes that "Spack packages must be loaded to use them," and Jarvis automates or guides the user through loading necessary modules (e.g., `spack load hermes`) before pipeline execution [cite: 1, 4].

### 4.3 Python API and CLI
Jarvis provides dual interfaces for interaction:
*   **CLI:** A command-line interface for interactive use (e.g., `jarvis pipeline create`, `jarvis resource-graph build`) [cite: 2, 4].
*   **Python API:** The core logic is implemented in Python, and `jarvis-util` exposes functions to execute binaries and collect output programmatically [cite: 1]. This allows researchers to write complex "recipes" or scripts that define dynamic pipelines based on runtime conditions.

### 4.4 Container Orchestration and Future Directions
While the user's query specifically asks about "container orchestration," the provided core documentation for Jarvis-CD focuses primarily on bare-metal deployment (using MPI and Spack) [cite: 1, 2]. However, broader context from the **IOWarp** and **LABIOS** projects (of which Jarvis is a component) indicates support for containerized workflows.
*   **Container Support:** The LABIOS project page lists "Container support" under its integration ecosystem, alongside Jarvis automation [cite: 5]. This suggests that while Jarvis's *mechanism* is process-based, it can orchestrate container runtimes (like Singularity/Apptainer or Docker) as "Application Pkgs."
*   **Cloud Models:** The PDSW'24 workshop, where Jarvis was presented, explicitly lists "Cloud and Container-Based Models" as a topic of interest [cite: 6].
*   **Agentic Workflows:** Recent work involves using AI agents to orchestrate workflows via Jarvis, comparing different LLMs (e.g., GPT-4o, Claude) in generating deployment configurations [cite: 7]. This evolution points toward Jarvis becoming a higher-level orchestrator that can manage both traditional MPI jobs and modern containerized microservices, potentially bridging the gap between static HPC schedulers (Slurm) and dynamic cloud orchestrators.

---

## 5. Conclusion

The Jarvis framework represents a "DevOps for HPC" approach. It addresses the rigidity of traditional HPC scripts by introducing structured abstractions—Pipelines, Packages, and Resource Graphs. Its architecture is specifically tuned for the data-centric needs of scientific computing, providing native support for complex storage hierarchies and interception-based I/O middleware. While its roots are in bare-metal MPI deployment, its integration into the IOWarp ecosystem positions it as a flexible tool capable of supporting emerging containerized and AI-driven workflows in supercomputing environments.

---

## References

### Publications
[cite: 1] "Jarvis-CD Index" (Gnosis Research Center). GRC Documentation. https://grc.iit.edu/docs/jarvis/jarvis-cd/index/
[cite: 2] "Jarvis-CD Design & Motivation" (Gnosis Research Center). GRC Documentation. https://grc.iit.edu/docs/jarvis/jarvis-cd/design-motivation/
[cite: 1] "Jarvis-CD Installation & Setup" (Gnosis Research Center). GRC Documentation. https://grc.iit.edu/docs/jarvis/jarvis-cd/index/
[cite: 3] "Platform Plugins Interface (PPI)" (IOWarp). GitHub Repository. https://github.com/iowarp/platform-plugins-interface
[cite: 4] "Gray Scott Benchmark with Jarvis" (SCS Lab). GitHub Repository. https://github.com/scs-lab/jarvis-cd/blob/master/builtin/builtin/gray_scott/README.md
[cite: 8] "Jarvis-CD Repository" (GRC IIT). GitHub Repository. https://github.com/grc-iit/jarvis-cd
[cite: 6] "PDSW 2025 Website" (PDSW). PDSW Conference Site. https://www.pdsw.org/
[cite: 7] "IOWarp Project Page" (Gnosis Research Center). GRC Research. https://grc.iit.edu/research/projects/iowarp/
[cite: 5] "LABIOS Project Page" (Gnosis Research Center). GRC Research. https://github.com/grc-iit/labios
[cite: 9] "Jaime Cernuda Profile" (Jaime Cernuda). Personal Website. https://jcernuda.com/
[cite: 10] "Jaime Cernuda Member Page" (Gnosis Research Center). GRC Members. https://grc.iit.edu/members/jaime-cernuda/
[cite: 11] "GRC Publications" (Gnosis Research Center). GRC Publications. https://grc.iit.edu/publications/
[cite: 12] "LABIOS Research Project" (Gnosis Research Center). GRC Research. https://grc.iit.edu/research/projects/labios/

**Sources:**
1. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEaBQbvxaupp6elrJWUxCbtJttZZQTjSpCFeYBjzDW0r1t_3Sx3LPE_WbdbHhYtaqiEmBv-S-Wfzk8YQS4e5vXH4PPAWTYvA2OTSsWbOIc33Rvgwus4rIMV4lnsmPxZTyJMRZs3xw==)
2. [iit.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFsdVNsAmM1nM5FZVjnrB-V8cC91IoVD3wnIkWDP2g3e6oWcREHL3vGu6UdMKxbv687UjYIOQuId7peaZGqhybvLM9oLJliLZHypsYhgLn604J90uUqmWIECpyK5TEvJMtJ6FW0pBZCgFRb0EM7-tM7bQ==)
3. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGT_gxlt_zdt1_mTkQmYqQ0Mz8eTGj3IdBRYPSDNvOjDIKod3bYHprULcYfFZ-1TAZmShIhzHZzH3VBPRud9Rvt9jgWDSDVVl4dSiz6YlphskE95PsbeJ9jJc9veMOPUntk-bqZDGVu4pA=)
4. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEI_QfeDeIZNp6A05PviglJO_1j3w1uPErWhVE83vTz7QPferWFQEfMlnjA4DR04OgMMnQXbFtPMYBPNiD3AWtqsZcfoJAJmBac5CJnTAIyqalAifGzD9QJnVphhpGv73xTyCzRXDCLOhF8kWYyVvQzu8nSRJLj4VTIUYiIILI55fVZoZknPvYNxqI=)
5. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF7pI3RL8NOdNxQ6eBvqxpLn-UtKsGB-xt1L3ydorzGXqGjT08ztFFKiQD7N5DnCSOzr1sdZO5Q8FnKovzGWo3fRxcNIOr8LePzt6OX_xFHbzzdlmerOQ==)
6. [pdsw.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQErieP_04_KLlyXgakW5dqn5wnbqKWbhIk3eYSieGN6hfpfwiDjkKNOQpS0aThwM6bT4CV6AcMjnrwgMeD2DCDh-eDRVBNYcwcsug==)
7. [iit.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEohdq_GtXd_lxpGDNWF9RQgRg9P4rfqYACMVC8nCrA31u6j-KoctiHmKBKsa7YS1XYSUmT5QaP6oyjwwJ26g_uVjFy5b-sZkGIrA4JmXvPeKajg7P5MXTnYb9EEGkrKJmSbw==)
8. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJzfth72R7be1-EfWcA303m1t4n56m3m80Di903-REVr8-DXss3Fc9ROqbS6lrbnJaM12rTjXZQn6AwDU3BJccjIQO-itXqVoBM9Dupuj2w8INjyRdhfklCg==)
9. [jcernuda.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGRcB2EdlZtoDfzDRX4Kw2uaFMLi5UtMDZHCYQC2cLET12acArPZGuZrbDa20BN5hs8HO8gEJjuo5ij2HZvbErN3RH0ErQLcOeLHg==)
10. [iit.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG12Qu2672d81o1EouVzdZxWQ-WdUyTnyJz4kGslj_2aFcmzD6QnJgvLyltvGkGIYL9V_eJE3fw10XDqcKcFUJFvACvidm1GbUlyRLm4qz5xM2DDCqB9AWtDpMzgVGW2w==)
11. [iit.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE4HzjF-ibSYlZqvrJn5mkqdcWWB8KC4kzdbYVOZvGgMj5M_BfpLXBj5UT7HXPkeV5sd2sW1OpWsd--oJDtd51FJ2K2Tj2aSv2hz8PyJSyL9v0S1dxdIg==)
12. [iit.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFFZsV0pgeaQFSzRFVrKJrqHwnNmv_2UijW4H8PnnYldrVVoVyXnovBA901CIKRKcDaKPfSpSRhNFm0foQ9e9qONaGOstGnoBxm1JoFcEs4ESt-WrnTp11Mek2r0Lg9zdQVYg==)
