"""
DORIAN COMPUTER SCIENCE & ENGINEERING DOMAIN
=============================================

Comprehensive computing knowledge for Dorian Core.

Includes:
1. Hardware (CPUs, GPUs, RAM, storage, architectures)
2. Software (operating systems, languages, paradigms)
3. Algorithms and data structures
4. Networking and distributed systems
5. Artificial Intelligence and Machine Learning
6. Computer architecture and organization

Author: Joseph + Claude
Date: 2026-01-25
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# COMPUTER SCIENCE CATEGORIES
# =============================================================================

CS_CATEGORIES = [
    # =========================================================================
    # FOUNDATIONAL CONCEPTS
    # =========================================================================
    # Core abstractions
    ("computation", "process", "The execution of an algorithm"),
    ("information", "abstract", "Data with meaning"),
    ("data", "abstract", "Raw facts and figures"),
    ("program", "abstract", "A sequence of instructions"),
    ("algorithm", "abstract", "A finite sequence of well-defined instructions"),
    ("protocol", "abstract", "A set of rules for communication"),
    # Computational models
    (
        "computational_model",
        "mathematical_structure",
        "An abstract model of computation",
    ),
    ("turing_machine", "computational_model", "A theoretical model of computation"),
    ("finite_automaton", "computational_model", "A machine with finite states"),
    ("pushdown_automaton", "computational_model", "An automaton with a stack"),
    ("lambda_calculus", "computational_model", "A formal system for computation"),
    ("cellular_automaton", "computational_model", "A grid of cells with update rules"),
    # =========================================================================
    # HARDWARE - PROCESSORS
    # =========================================================================
    # Processor types
    ("processor", "physical_object", "A device that executes instructions"),
    ("cpu", "processor", "Central Processing Unit - general purpose processor"),
    ("gpu", "processor", "Graphics Processing Unit - parallel processor"),
    ("tpu", "processor", "Tensor Processing Unit - AI accelerator"),
    ("npu", "processor", "Neural Processing Unit - neural network accelerator"),
    ("dsp", "processor", "Digital Signal Processor"),
    ("fpga", "processor", "Field Programmable Gate Array"),
    ("asic", "processor", "Application Specific Integrated Circuit"),
    ("microcontroller", "processor", "A small computer on a single chip"),
    ("coprocessor", "processor", "A processor that assists the main CPU"),
    # CPU components
    ("cpu_component", "physical_object", "A component of a CPU"),
    ("alu", "cpu_component", "Arithmetic Logic Unit"),
    ("fpu", "cpu_component", "Floating Point Unit"),
    ("control_unit", "cpu_component", "Unit that directs CPU operations"),
    ("register", "cpu_component", "Fast storage within CPU"),
    ("program_counter", "register", "Register holding next instruction address"),
    ("instruction_register", "register", "Register holding current instruction"),
    ("accumulator", "register", "Register for arithmetic results"),
    ("stack_pointer", "register", "Register pointing to top of stack"),
    ("cpu_cache", "cpu_component", "Fast memory close to CPU"),
    ("l1_cache", "cpu_cache", "Level 1 cache - fastest, smallest"),
    ("l2_cache", "cpu_cache", "Level 2 cache - larger, slower than L1"),
    ("l3_cache", "cpu_cache", "Level 3 cache - shared across cores"),
    ("branch_predictor", "cpu_component", "Unit that predicts branch outcomes"),
    ("instruction_decoder", "cpu_component", "Unit that decodes instructions"),
    ("execution_unit", "cpu_component", "Unit that executes instructions"),
    ("load_store_unit", "cpu_component", "Unit handling memory operations"),
    ("memory_controller", "cpu_component", "Unit managing memory access"),
    # CPU architectures
    ("cpu_architecture", "structure", "The design of a CPU"),
    ("x86", "cpu_architecture", "Intel/AMD architecture"),
    ("x86_64", "cpu_architecture", "64-bit extension of x86"),
    ("arm", "cpu_architecture", "ARM architecture - mobile/embedded"),
    ("arm64", "cpu_architecture", "64-bit ARM architecture"),
    ("risc_v", "cpu_architecture", "Open source RISC architecture"),
    ("mips", "cpu_architecture", "MIPS architecture"),
    ("powerpc", "cpu_architecture", "PowerPC architecture"),
    ("sparc", "cpu_architecture", "SPARC architecture"),
    # Instruction set types
    ("instruction_set", "structure", "The set of instructions a CPU can execute"),
    ("cisc", "instruction_set", "Complex Instruction Set Computing"),
    ("risc", "instruction_set", "Reduced Instruction Set Computing"),
    ("vliw", "instruction_set", "Very Long Instruction Word"),
    ("simd", "instruction_set", "Single Instruction Multiple Data"),
    ("mimd", "instruction_set", "Multiple Instruction Multiple Data"),
    # =========================================================================
    # HARDWARE - GPU
    # =========================================================================
    ("gpu_component", "physical_object", "A component of a GPU"),
    ("shader_core", "gpu_component", "A processing unit in a GPU"),
    ("cuda_core", "shader_core", "NVIDIA CUDA processing core"),
    ("stream_processor", "shader_core", "AMD stream processor"),
    ("tensor_core", "gpu_component", "Matrix operation accelerator"),
    ("rt_core", "gpu_component", "Ray tracing core"),
    ("texture_unit", "gpu_component", "Unit for texture operations"),
    ("rasterizer", "gpu_component", "Unit converting vectors to pixels"),
    ("vram", "memory", "Video RAM - GPU memory"),
    ("gddr", "vram", "Graphics DDR memory"),
    ("gddr6", "gddr", "GDDR6 memory standard"),
    ("gddr6x", "gddr", "GDDR6X memory standard"),
    ("hbm", "vram", "High Bandwidth Memory"),
    ("hbm2", "hbm", "HBM2 memory standard"),
    ("hbm3", "hbm", "HBM3 memory standard"),
    # GPU architectures
    ("gpu_architecture", "structure", "The design of a GPU"),
    ("nvidia_architecture", "gpu_architecture", "NVIDIA GPU architecture"),
    ("ampere", "nvidia_architecture", "NVIDIA Ampere architecture"),
    ("hopper", "nvidia_architecture", "NVIDIA Hopper architecture"),
    ("ada_lovelace", "nvidia_architecture", "NVIDIA Ada Lovelace architecture"),
    ("blackwell", "nvidia_architecture", "NVIDIA Blackwell architecture"),
    ("amd_architecture", "gpu_architecture", "AMD GPU architecture"),
    ("rdna", "amd_architecture", "AMD RDNA architecture"),
    ("cdna", "amd_architecture", "AMD CDNA architecture"),
    # =========================================================================
    # HARDWARE - MEMORY
    # =========================================================================
    ("memory", "physical_object", "Storage for data"),
    ("volatile_memory", "memory", "Memory that loses data when powered off"),
    ("non_volatile_memory", "memory", "Memory that retains data without power"),
    # RAM types
    ("ram", "volatile_memory", "Random Access Memory"),
    ("dram", "ram", "Dynamic RAM - needs refreshing"),
    ("sram", "ram", "Static RAM - faster, no refresh needed"),
    ("sdram", "dram", "Synchronous DRAM"),
    ("ddr", "sdram", "Double Data Rate SDRAM"),
    ("ddr3", "ddr", "DDR3 memory standard"),
    ("ddr4", "ddr", "DDR4 memory standard"),
    ("ddr5", "ddr", "DDR5 memory standard"),
    ("lpddr", "ddr", "Low Power DDR for mobile"),
    ("ecc_ram", "ram", "Error Correcting Code RAM"),
    # Storage
    ("storage", "non_volatile_memory", "Persistent data storage"),
    ("hdd", "storage", "Hard Disk Drive - magnetic storage"),
    ("ssd", "storage", "Solid State Drive - flash storage"),
    ("nvme", "ssd", "NVMe SSD - PCIe connected"),
    ("sata_ssd", "ssd", "SATA connected SSD"),
    ("flash_memory", "non_volatile_memory", "Flash-based storage"),
    ("nand_flash", "flash_memory", "NAND flash memory"),
    ("slc", "nand_flash", "Single Level Cell flash"),
    ("mlc", "nand_flash", "Multi Level Cell flash"),
    ("tlc", "nand_flash", "Triple Level Cell flash"),
    ("qlc", "nand_flash", "Quad Level Cell flash"),
    ("optane", "non_volatile_memory", "Intel Optane memory"),
    # =========================================================================
    # HARDWARE - INTERCONNECTS
    # =========================================================================
    ("bus", "physical_object", "A communication pathway"),
    ("system_bus", "bus", "Main bus connecting CPU to memory"),
    ("pcie", "bus", "PCI Express - high speed serial bus"),
    ("pcie_4", "pcie", "PCIe 4.0 standard"),
    ("pcie_5", "pcie", "PCIe 5.0 standard"),
    ("usb", "bus", "Universal Serial Bus"),
    ("usb_3", "usb", "USB 3.x standard"),
    ("usb_4", "usb", "USB 4 standard"),
    ("thunderbolt", "bus", "High speed interface"),
    ("sata", "bus", "Serial ATA interface"),
    ("nvlink", "bus", "NVIDIA high-speed GPU interconnect"),
    ("infinity_fabric", "bus", "AMD interconnect technology"),
    # =========================================================================
    # HARDWARE - SYSTEMS
    # =========================================================================
    ("computer_system", "system", "A complete computing system"),
    ("motherboard", "physical_object", "Main circuit board"),
    ("chipset", "physical_object", "Set of chips managing data flow"),
    ("bios", "software", "Basic Input Output System"),
    ("uefi", "software", "Unified Extensible Firmware Interface"),
    ("power_supply", "physical_object", "Provides power to components"),
    ("cooling_system", "physical_object", "Removes heat from components"),
    # Computer types
    ("personal_computer", "computer_system", "A computer for individual use"),
    ("desktop", "personal_computer", "A stationary personal computer"),
    ("laptop", "personal_computer", "A portable computer"),
    ("workstation", "personal_computer", "High performance personal computer"),
    ("server", "computer_system", "A computer providing services"),
    ("mainframe", "computer_system", "A large powerful computer"),
    ("supercomputer", "computer_system", "An extremely powerful computer"),
    ("embedded_system", "computer_system", "A computer within a larger system"),
    ("iot_device", "embedded_system", "Internet of Things device"),
    ("cluster", "computer_system", "Multiple computers working together"),
    ("datacenter", "computer_system", "A facility housing many servers"),
    # =========================================================================
    # SOFTWARE - OPERATING SYSTEMS
    # =========================================================================
    ("software", "abstract", "Programs and data that run on hardware"),
    ("system_software", "software", "Software managing hardware resources"),
    ("application_software", "software", "Software for end users"),
    ("operating_system", "system_software", "Software managing computer resources"),
    ("kernel", "operating_system", "Core of the operating system"),
    ("monolithic_kernel", "kernel", "Kernel with all services in kernel space"),
    ("microkernel", "kernel", "Kernel with minimal services"),
    ("hybrid_kernel", "kernel", "Combination of monolithic and micro"),
    # OS types
    ("unix", "operating_system", "Unix operating system family"),
    ("linux", "unix", "Linux operating system"),
    ("linux_kernel", "kernel", "The Linux kernel"),
    ("ubuntu", "linux", "Ubuntu Linux distribution"),
    ("debian", "linux", "Debian Linux distribution"),
    ("fedora", "linux", "Fedora Linux distribution"),
    ("centos", "linux", "CentOS Linux distribution"),
    ("rhel", "linux", "Red Hat Enterprise Linux"),
    ("arch_linux", "linux", "Arch Linux distribution"),
    ("android", "linux", "Android mobile operating system"),
    ("bsd", "unix", "BSD operating system family"),
    ("freebsd", "bsd", "FreeBSD operating system"),
    ("macos", "bsd", "macOS operating system"),
    ("ios", "bsd", "iOS mobile operating system"),
    ("windows", "operating_system", "Microsoft Windows"),
    ("windows_nt", "kernel", "Windows NT kernel"),
    # OS components
    ("device_driver", "system_software", "Software controlling hardware"),
    ("file_system", "system_software", "System for organizing files"),
    ("ext4", "file_system", "Extended filesystem 4"),
    ("ntfs", "file_system", "NT File System"),
    ("apfs", "file_system", "Apple File System"),
    ("zfs", "file_system", "ZFS file system"),
    ("btrfs", "file_system", "B-tree file system"),
    ("scheduler", "system_software", "Component that schedules processes"),
    ("memory_manager", "system_software", "Component managing memory"),
    ("virtual_memory", "memory_manager", "Abstraction providing more memory"),
    ("paging", "virtual_memory", "Memory management using pages"),
    ("process", "computation", "An instance of a running program"),
    ("thread", "computation", "A unit of execution within a process"),
    # =========================================================================
    # SOFTWARE - PROGRAMMING LANGUAGES
    # =========================================================================
    ("programming_language", "abstract", "A language for writing programs"),
    # Language paradigms
    ("language_paradigm", "abstract", "A style of programming"),
    ("imperative", "language_paradigm", "Programming with statements"),
    ("declarative", "language_paradigm", "Programming with declarations"),
    ("functional", "language_paradigm", "Programming with functions"),
    ("object_oriented", "language_paradigm", "Programming with objects"),
    ("procedural", "language_paradigm", "Programming with procedures"),
    ("logic", "language_paradigm", "Programming with logical statements"),
    ("concurrent", "language_paradigm", "Programming with concurrent execution"),
    # Language types
    ("compiled_language", "programming_language", "Language compiled to machine code"),
    (
        "interpreted_language",
        "programming_language",
        "Language executed by interpreter",
    ),
    ("jit_compiled", "programming_language", "Just-in-time compiled language"),
    # Specific languages
    ("assembly", "programming_language", "Low-level programming language"),
    ("c", "compiled_language", "C programming language"),
    ("cpp", "compiled_language", "C++ programming language"),
    ("rust", "compiled_language", "Rust programming language"),
    ("go", "compiled_language", "Go programming language"),
    ("java", "jit_compiled", "Java programming language"),
    ("kotlin", "jit_compiled", "Kotlin programming language"),
    ("scala", "jit_compiled", "Scala programming language"),
    ("csharp", "jit_compiled", "C# programming language"),
    ("python", "interpreted_language", "Python programming language"),
    ("javascript", "jit_compiled", "JavaScript programming language"),
    ("typescript", "jit_compiled", "TypeScript programming language"),
    ("ruby", "interpreted_language", "Ruby programming language"),
    ("php", "interpreted_language", "PHP programming language"),
    ("perl", "interpreted_language", "Perl programming language"),
    ("swift", "compiled_language", "Swift programming language"),
    ("haskell", "compiled_language", "Haskell functional language"),
    ("lisp", "interpreted_language", "Lisp programming language"),
    ("prolog", "interpreted_language", "Prolog logic language"),
    ("sql", "declarative", "Structured Query Language"),
    ("cuda", "programming_language", "NVIDIA GPU programming"),
    ("opencl", "programming_language", "Open Computing Language"),
    # =========================================================================
    # DATA STRUCTURES
    # =========================================================================
    ("data_structure", "abstract", "A way of organizing data"),
    # Linear structures
    ("linear_structure", "data_structure", "A sequential data structure"),
    ("array", "linear_structure", "A contiguous sequence of elements"),
    ("dynamic_array", "array", "A resizable array"),
    ("linked_list", "linear_structure", "Elements linked by pointers"),
    ("singly_linked_list", "linked_list", "Links in one direction"),
    ("doubly_linked_list", "linked_list", "Links in both directions"),
    ("stack", "linear_structure", "Last-in-first-out structure"),
    ("queue", "linear_structure", "First-in-first-out structure"),
    ("deque", "linear_structure", "Double-ended queue"),
    ("priority_queue", "queue", "Queue ordered by priority"),
    # Trees
    ("tree", "data_structure", "A hierarchical structure"),
    ("binary_tree", "tree", "Tree with at most two children"),
    ("binary_search_tree", "binary_tree", "Ordered binary tree"),
    ("avl_tree", "binary_search_tree", "Self-balancing BST"),
    ("red_black_tree", "binary_search_tree", "Self-balancing BST with colors"),
    ("b_tree", "tree", "Self-balancing tree for databases"),
    ("b_plus_tree", "b_tree", "B-tree variant for filesystems"),
    ("trie", "tree", "Tree for string prefixes"),
    ("heap", "tree", "Tree satisfying heap property"),
    ("min_heap", "heap", "Heap with minimum at root"),
    ("max_heap", "heap", "Heap with maximum at root"),
    ("segment_tree", "tree", "Tree for range queries"),
    ("fenwick_tree", "tree", "Binary indexed tree"),
    # Graphs
    ("graph", "data_structure", "Nodes connected by edges"),
    ("directed_graph", "graph", "Graph with directed edges"),
    ("undirected_graph", "graph", "Graph with undirected edges"),
    ("weighted_graph", "graph", "Graph with edge weights"),
    ("dag", "directed_graph", "Directed acyclic graph"),
    ("adjacency_list", "graph", "Graph as list of neighbors"),
    ("adjacency_matrix", "graph", "Graph as connection matrix"),
    # Hash-based
    ("hash_table", "data_structure", "Structure using hash functions"),
    ("hash_map", "hash_table", "Key-value hash table"),
    ("hash_set", "hash_table", "Set using hashing"),
    ("bloom_filter", "data_structure", "Probabilistic set membership"),
    # =========================================================================
    # ALGORITHMS
    # =========================================================================
    ("algorithm_type", "abstract", "A category of algorithms"),
    # Sorting
    ("sorting_algorithm", "algorithm_type", "Algorithm for ordering elements"),
    ("comparison_sort", "sorting_algorithm", "Sort using comparisons"),
    ("quicksort", "comparison_sort", "Divide and conquer sort"),
    ("mergesort", "comparison_sort", "Divide and conquer stable sort"),
    ("heapsort", "comparison_sort", "Sort using a heap"),
    ("insertion_sort", "comparison_sort", "Sort by insertion"),
    ("bubble_sort", "comparison_sort", "Sort by adjacent swaps"),
    ("timsort", "comparison_sort", "Hybrid sort used in Python"),
    ("radix_sort", "sorting_algorithm", "Non-comparison integer sort"),
    ("counting_sort", "sorting_algorithm", "Sort by counting occurrences"),
    # Searching
    ("search_algorithm", "algorithm_type", "Algorithm for finding elements"),
    ("linear_search", "search_algorithm", "Sequential search"),
    ("binary_search", "search_algorithm", "Search in sorted array"),
    ("hash_lookup", "search_algorithm", "Search using hash table"),
    ("interpolation_search", "search_algorithm", "Search using interpolation"),
    # Graph algorithms
    ("graph_algorithm", "algorithm_type", "Algorithm on graphs"),
    ("bfs", "graph_algorithm", "Breadth-first search"),
    ("dfs", "graph_algorithm", "Depth-first search"),
    ("dijkstra", "graph_algorithm", "Shortest path algorithm"),
    ("bellman_ford", "graph_algorithm", "Shortest path with negative weights"),
    ("floyd_warshall", "graph_algorithm", "All-pairs shortest path"),
    ("a_star", "graph_algorithm", "Heuristic pathfinding"),
    ("prims", "graph_algorithm", "Minimum spanning tree"),
    ("kruskals", "graph_algorithm", "Minimum spanning tree"),
    ("topological_sort", "graph_algorithm", "Ordering of DAG nodes"),
    # String algorithms
    ("string_algorithm", "algorithm_type", "Algorithm on strings"),
    ("kmp", "string_algorithm", "Knuth-Morris-Pratt pattern matching"),
    ("rabin_karp", "string_algorithm", "Pattern matching with hashing"),
    ("boyer_moore", "string_algorithm", "Efficient pattern matching"),
    ("levenshtein", "string_algorithm", "Edit distance algorithm"),
    ("suffix_array", "string_algorithm", "Array of sorted suffixes"),
    # Dynamic programming
    ("dynamic_programming", "algorithm_type", "Optimization by subproblems"),
    ("memoization", "dynamic_programming", "Caching subproblem results"),
    ("tabulation", "dynamic_programming", "Bottom-up DP"),
    # =========================================================================
    # COMPLEXITY THEORY
    # =========================================================================
    ("complexity_class", "abstract", "A class of computational problems"),
    ("p", "complexity_class", "Problems solvable in polynomial time"),
    ("np", "complexity_class", "Problems verifiable in polynomial time"),
    ("np_complete", "complexity_class", "Hardest problems in NP"),
    ("np_hard", "complexity_class", "At least as hard as NP-complete"),
    ("pspace", "complexity_class", "Problems solvable with polynomial space"),
    ("exptime", "complexity_class", "Problems solvable in exponential time"),
    ("time_complexity", "property", "How runtime grows with input"),
    ("space_complexity", "property", "How memory grows with input"),
    ("big_o", "mathematical_object", "Upper bound notation"),
    ("big_omega", "mathematical_object", "Lower bound notation"),
    ("big_theta", "mathematical_object", "Tight bound notation"),
    # =========================================================================
    # NETWORKING
    # =========================================================================
    ("network", "system", "Connected computing devices"),
    ("computer_network", "network", "Network of computers"),
    ("lan", "computer_network", "Local Area Network"),
    ("wan", "computer_network", "Wide Area Network"),
    ("internet", "wan", "Global network of networks"),
    ("intranet", "lan", "Private organizational network"),
    # Network layers (OSI)
    ("network_layer", "abstract", "A layer in network architecture"),
    ("physical_layer", "network_layer", "Layer 1 - physical transmission"),
    ("data_link_layer", "network_layer", "Layer 2 - node to node"),
    ("network_layer_3", "network_layer", "Layer 3 - routing"),
    ("transport_layer", "network_layer", "Layer 4 - end to end"),
    ("session_layer", "network_layer", "Layer 5 - sessions"),
    ("presentation_layer", "network_layer", "Layer 6 - data format"),
    ("application_layer", "network_layer", "Layer 7 - applications"),
    # Protocols
    ("network_protocol", "protocol", "A networking protocol"),
    ("tcp", "network_protocol", "Transmission Control Protocol"),
    ("udp", "network_protocol", "User Datagram Protocol"),
    ("ip", "network_protocol", "Internet Protocol"),
    ("ipv4", "ip", "IP version 4"),
    ("ipv6", "ip", "IP version 6"),
    ("http", "network_protocol", "Hypertext Transfer Protocol"),
    ("https", "http", "HTTP Secure"),
    ("ftp", "network_protocol", "File Transfer Protocol"),
    ("ssh", "network_protocol", "Secure Shell"),
    ("dns", "network_protocol", "Domain Name System"),
    ("dhcp", "network_protocol", "Dynamic Host Configuration Protocol"),
    ("tls", "network_protocol", "Transport Layer Security"),
    ("websocket", "network_protocol", "Full-duplex communication protocol"),
    ("grpc", "network_protocol", "Google RPC protocol"),
    # Network hardware
    ("network_device", "physical_object", "Hardware for networking"),
    ("router", "network_device", "Device routing packets"),
    ("switch", "network_device", "Device switching frames"),
    ("hub", "network_device", "Device broadcasting data"),
    ("modem", "network_device", "Modulator-demodulator"),
    ("firewall", "network_device", "Security gateway"),
    ("load_balancer", "network_device", "Distributes network traffic"),
    ("nic", "network_device", "Network Interface Card"),
    # =========================================================================
    # DATABASES
    # =========================================================================
    ("database", "software", "Organized collection of data"),
    ("dbms", "software", "Database Management System"),
    # Database types
    ("relational_database", "database", "Database using relations/tables"),
    ("nosql_database", "database", "Non-relational database"),
    ("document_database", "nosql_database", "Stores documents"),
    ("key_value_store", "nosql_database", "Simple key-value pairs"),
    ("graph_database", "nosql_database", "Stores graph structures"),
    ("column_store", "nosql_database", "Column-oriented storage"),
    ("time_series_database", "database", "Optimized for time series"),
    ("vector_database", "database", "Optimized for vector search"),
    # Specific databases
    ("postgresql", "relational_database", "PostgreSQL database"),
    ("mysql", "relational_database", "MySQL database"),
    ("sqlite", "relational_database", "SQLite embedded database"),
    ("oracle", "relational_database", "Oracle database"),
    ("mongodb", "document_database", "MongoDB document store"),
    ("redis", "key_value_store", "Redis in-memory store"),
    ("cassandra", "column_store", "Apache Cassandra"),
    ("neo4j", "graph_database", "Neo4j graph database"),
    ("elasticsearch", "document_database", "Elasticsearch search engine"),
    ("pinecone", "vector_database", "Pinecone vector database"),
    ("milvus", "vector_database", "Milvus vector database"),
    ("faiss", "vector_database", "Facebook AI Similarity Search"),
    # Database concepts
    ("acid", "property", "Atomicity Consistency Isolation Durability"),
    ("transaction", "computation", "A unit of database work"),
    ("index", "data_structure", "Structure for fast lookup"),
    ("query", "computation", "A request for data"),
    ("join", "query", "Combining tables"),
    ("normalization", "process", "Organizing database structure"),
    # =========================================================================
    # DISTRIBUTED SYSTEMS
    # =========================================================================
    ("distributed_system", "system", "System spanning multiple computers"),
    ("distributed_computing", "computation", "Computing across multiple nodes"),
    # Concepts
    ("consistency", "property", "All nodes see same data"),
    ("availability", "property", "System always responds"),
    ("partition_tolerance", "property", "System works despite network issues"),
    ("cap_theorem", "theorem", "Can only have 2 of CAP"),
    ("eventual_consistency", "consistency", "Consistency over time"),
    ("strong_consistency", "consistency", "Immediate consistency"),
    # Patterns
    ("microservices", "structure", "Architecture of small services"),
    ("monolith", "structure", "Single unified application"),
    ("soa", "structure", "Service Oriented Architecture"),
    ("message_queue", "data_structure", "Asynchronous message passing"),
    ("event_sourcing", "pattern", "Storing events not state"),
    ("cqrs", "pattern", "Command Query Responsibility Segregation"),
    # Technologies
    ("kubernetes", "software", "Container orchestration"),
    ("docker", "software", "Container platform"),
    ("container", "software", "Isolated execution environment"),
    ("kafka", "message_queue", "Apache Kafka streaming"),
    ("rabbitmq", "message_queue", "RabbitMQ message broker"),
    ("zookeeper", "software", "Distributed coordination"),
    ("etcd", "key_value_store", "Distributed key-value store"),
    ("consul", "software", "Service mesh and discovery"),
    # =========================================================================
    # ARTIFICIAL INTELLIGENCE
    # =========================================================================
    ("artificial_intelligence", "field", "The field of AI"),
    ("ai_system", "software", "A system exhibiting intelligence"),
    # AI types
    ("narrow_ai", "ai_system", "AI for specific tasks"),
    ("general_ai", "ai_system", "Human-level AI"),
    ("superintelligence", "ai_system", "AI surpassing human intelligence"),
    # AI approaches
    ("symbolic_ai", "ai_system", "AI using symbols and rules"),
    ("connectionist_ai", "ai_system", "AI using neural networks"),
    ("hybrid_ai", "ai_system", "Combining symbolic and connectionist"),
    ("neuro_symbolic", "hybrid_ai", "Neural-symbolic integration"),
    # =========================================================================
    # MACHINE LEARNING
    # =========================================================================
    ("machine_learning", "artificial_intelligence", "Learning from data"),
    ("ml_algorithm", "algorithm", "A machine learning algorithm"),
    # Learning types
    ("supervised_learning", "machine_learning", "Learning from labeled data"),
    ("unsupervised_learning", "machine_learning", "Learning from unlabeled data"),
    ("reinforcement_learning", "machine_learning", "Learning from rewards"),
    ("semi_supervised", "machine_learning", "Mix of labeled and unlabeled"),
    ("self_supervised", "machine_learning", "Creating own labels"),
    ("transfer_learning", "machine_learning", "Reusing learned knowledge"),
    ("meta_learning", "machine_learning", "Learning to learn"),
    ("few_shot_learning", "machine_learning", "Learning from few examples"),
    ("zero_shot_learning", "machine_learning", "Learning without examples"),
    ("continual_learning", "machine_learning", "Learning over time"),
    ("federated_learning", "machine_learning", "Distributed learning"),
    # Supervised algorithms
    ("classification", "supervised_learning", "Predicting categories"),
    ("regression", "supervised_learning", "Predicting continuous values"),
    ("linear_regression", "regression", "Linear model for regression"),
    ("logistic_regression", "classification", "Linear model for classification"),
    ("decision_tree", "ml_algorithm", "Tree-based decisions"),
    ("random_forest", "ml_algorithm", "Ensemble of decision trees"),
    ("gradient_boosting", "ml_algorithm", "Sequential ensemble method"),
    ("xgboost", "gradient_boosting", "Extreme Gradient Boosting"),
    ("lightgbm", "gradient_boosting", "Light Gradient Boosting"),
    ("svm", "ml_algorithm", "Support Vector Machine"),
    ("knn", "ml_algorithm", "K-Nearest Neighbors"),
    ("naive_bayes", "ml_algorithm", "Probabilistic classifier"),
    # Unsupervised algorithms
    ("clustering", "unsupervised_learning", "Grouping similar items"),
    ("kmeans", "clustering", "K-means clustering"),
    ("dbscan", "clustering", "Density-based clustering"),
    ("hierarchical_clustering", "clustering", "Tree of clusters"),
    ("dimensionality_reduction", "unsupervised_learning", "Reducing dimensions"),
    ("pca", "dimensionality_reduction", "Principal Component Analysis"),
    ("tsne", "dimensionality_reduction", "t-SNE visualization"),
    ("umap", "dimensionality_reduction", "UMAP embedding"),
    ("autoencoder", "unsupervised_learning", "Learning compressed representations"),
    # =========================================================================
    # DEEP LEARNING
    # =========================================================================
    ("deep_learning", "machine_learning", "Learning with deep neural networks"),
    ("neural_network", "ml_algorithm", "Network of artificial neurons"),
    # Neural network types
    ("feedforward_network", "neural_network", "Connections in one direction"),
    ("mlp", "feedforward_network", "Multi-Layer Perceptron"),
    ("cnn", "neural_network", "Convolutional Neural Network"),
    ("rnn", "neural_network", "Recurrent Neural Network"),
    ("lstm", "rnn", "Long Short-Term Memory"),
    ("gru", "rnn", "Gated Recurrent Unit"),
    ("transformer", "neural_network", "Attention-based architecture"),
    ("gan", "neural_network", "Generative Adversarial Network"),
    ("vae", "neural_network", "Variational Autoencoder"),
    ("diffusion_model", "neural_network", "Denoising diffusion model"),
    ("graph_neural_network", "neural_network", "Neural network on graphs"),
    ("capsule_network", "neural_network", "Network with capsules"),
    # Network components
    ("neuron", "mathematical_object", "A computational unit"),
    ("activation_function", "function", "Non-linearity in networks"),
    ("relu", "activation_function", "Rectified Linear Unit"),
    ("sigmoid", "activation_function", "Logistic function"),
    ("tanh", "activation_function", "Hyperbolic tangent"),
    ("softmax", "activation_function", "Probability distribution"),
    ("gelu", "activation_function", "Gaussian Error Linear Unit"),
    ("layer", "structure", "A level in a neural network"),
    ("dense_layer", "layer", "Fully connected layer"),
    ("conv_layer", "layer", "Convolutional layer"),
    ("pooling_layer", "layer", "Downsampling layer"),
    ("dropout_layer", "layer", "Regularization layer"),
    ("batch_norm", "layer", "Batch normalization"),
    ("layer_norm", "layer", "Layer normalization"),
    ("attention", "layer", "Attention mechanism"),
    ("self_attention", "attention", "Attention over same sequence"),
    ("cross_attention", "attention", "Attention between sequences"),
    ("multi_head_attention", "attention", "Multiple attention heads"),
    # Training
    ("backpropagation", "algorithm", "Training neural networks"),
    ("gradient_descent", "algorithm", "Optimization by gradients"),
    ("sgd", "gradient_descent", "Stochastic Gradient Descent"),
    ("adam", "gradient_descent", "Adaptive Moment Estimation"),
    ("adamw", "adam", "Adam with weight decay"),
    ("rmsprop", "gradient_descent", "Root Mean Square Propagation"),
    ("learning_rate", "parameter", "Step size for optimization"),
    ("loss_function", "function", "Function to minimize"),
    ("cross_entropy", "loss_function", "Cross-entropy loss"),
    ("mse", "loss_function", "Mean Squared Error"),
    ("regularization", "technique", "Preventing overfitting"),
    ("l1_regularization", "regularization", "Lasso regularization"),
    ("l2_regularization", "regularization", "Ridge regularization"),
    ("dropout", "regularization", "Randomly dropping units"),
    ("early_stopping", "regularization", "Stopping before overfitting"),
    # =========================================================================
    # LARGE LANGUAGE MODELS
    # =========================================================================
    ("language_model", "neural_network", "Model of language"),
    ("llm", "language_model", "Large Language Model"),
    # Architectures
    ("gpt", "transformer", "Generative Pre-trained Transformer"),
    ("gpt_3", "gpt", "GPT-3 model"),
    ("gpt_4", "gpt", "GPT-4 model"),
    ("bert", "transformer", "Bidirectional Encoder Representations"),
    ("t5", "transformer", "Text-to-Text Transfer Transformer"),
    ("llama", "transformer", "Meta's LLaMA model"),
    ("claude", "transformer", "Anthropic's Claude model"),
    ("palm", "transformer", "Google's PaLM model"),
    ("gemini", "transformer", "Google's Gemini model"),
    ("mistral", "transformer", "Mistral AI model"),
    # LLM concepts
    ("tokenization", "process", "Breaking text into tokens"),
    ("embedding", "vector", "Dense vector representation"),
    ("word_embedding", "embedding", "Vector for a word"),
    ("positional_encoding", "embedding", "Encoding position in sequence"),
    ("context_window", "parameter", "Maximum input length"),
    ("prompt", "data", "Input to a language model"),
    ("completion", "data", "Output from a language model"),
    ("fine_tuning", "training", "Adapting pre-trained model"),
    ("rlhf", "training", "Reinforcement Learning from Human Feedback"),
    ("dpo", "training", "Direct Preference Optimization"),
    ("instruction_tuning", "fine_tuning", "Training to follow instructions"),
    ("in_context_learning", "capability", "Learning from examples in prompt"),
    ("chain_of_thought", "technique", "Step-by-step reasoning"),
    ("retrieval_augmented", "technique", "Augmenting with retrieved info"),
    ("rag", "retrieval_augmented", "Retrieval Augmented Generation"),
    # =========================================================================
    # COMPUTER VISION
    # =========================================================================
    ("computer_vision", "artificial_intelligence", "Visual understanding"),
    ("image_classification", "computer_vision", "Categorizing images"),
    ("object_detection", "computer_vision", "Finding objects in images"),
    ("semantic_segmentation", "computer_vision", "Pixel-wise classification"),
    ("instance_segmentation", "computer_vision", "Distinguishing object instances"),
    ("image_generation", "computer_vision", "Creating images"),
    ("face_recognition", "computer_vision", "Identifying faces"),
    ("ocr", "computer_vision", "Optical Character Recognition"),
    ("pose_estimation", "computer_vision", "Detecting body pose"),
    # Vision models
    ("resnet", "cnn", "Residual Network"),
    ("vgg", "cnn", "VGG Network"),
    ("efficientnet", "cnn", "Efficient Network"),
    ("yolo", "cnn", "You Only Look Once detector"),
    ("vit", "transformer", "Vision Transformer"),
    ("clip", "neural_network", "Contrastive Language-Image Pre-training"),
    ("stable_diffusion", "diffusion_model", "Text-to-image generation"),
    ("dall_e", "diffusion_model", "OpenAI image generation"),
    ("midjourney", "diffusion_model", "Midjourney image generation"),
    # =========================================================================
    # NLP
    # =========================================================================
    ("nlp", "artificial_intelligence", "Natural Language Processing"),
    ("text_classification", "nlp", "Categorizing text"),
    ("sentiment_analysis", "text_classification", "Detecting sentiment"),
    ("named_entity_recognition", "nlp", "Finding named entities"),
    ("machine_translation", "nlp", "Translating languages"),
    ("question_answering", "nlp", "Answering questions"),
    ("text_summarization", "nlp", "Summarizing text"),
    ("text_generation", "nlp", "Generating text"),
    ("speech_recognition", "nlp", "Converting speech to text"),
    ("text_to_speech", "nlp", "Converting text to speech"),
    # =========================================================================
    # AI INFRASTRUCTURE
    # =========================================================================
    ("ml_framework", "software", "Framework for machine learning"),
    ("pytorch", "ml_framework", "PyTorch framework"),
    ("tensorflow", "ml_framework", "TensorFlow framework"),
    ("jax", "ml_framework", "JAX framework"),
    ("keras", "ml_framework", "Keras high-level API"),
    ("scikit_learn", "ml_framework", "Scikit-learn library"),
    ("huggingface", "ml_framework", "Hugging Face transformers"),
    ("ml_ops", "practice", "ML operations"),
    ("model_serving", "ml_ops", "Deploying ML models"),
    ("model_monitoring", "ml_ops", "Monitoring model performance"),
    ("feature_store", "software", "Managing ML features"),
    ("experiment_tracking", "ml_ops", "Tracking ML experiments"),
    ("mlflow", "software", "ML lifecycle platform"),
    ("wandb", "software", "Weights & Biases tracking"),
    ("ray", "software", "Distributed computing framework"),
    ("dask", "software", "Parallel computing library"),
    ("spark", "software", "Apache Spark distributed computing"),
    # Training infrastructure
    ("gpu_cluster", "cluster", "Cluster of GPUs"),
    ("tpu_pod", "cluster", "Google TPU cluster"),
    ("data_parallelism", "technique", "Parallelizing across data"),
    ("model_parallelism", "technique", "Parallelizing across model"),
    ("pipeline_parallelism", "technique", "Parallelizing across layers"),
    ("tensor_parallelism", "model_parallelism", "Parallelizing tensors"),
    ("mixed_precision", "technique", "Using multiple precisions"),
    ("fp16", "data_type", "16-bit floating point"),
    ("bf16", "data_type", "Brain floating point 16"),
    ("fp32", "data_type", "32-bit floating point"),
    ("int8", "data_type", "8-bit integer"),
    ("quantization", "technique", "Reducing precision"),
    ("pruning", "technique", "Removing network connections"),
    ("distillation", "technique", "Compressing models"),
    # =========================================================================
    # SECURITY
    # =========================================================================
    ("security", "field", "Protecting systems and data"),
    ("cryptography", "security", "Secure communication"),
    ("encryption", "cryptography", "Making data unreadable"),
    ("symmetric_encryption", "encryption", "Same key for encrypt/decrypt"),
    ("asymmetric_encryption", "encryption", "Public/private key pairs"),
    ("aes", "symmetric_encryption", "Advanced Encryption Standard"),
    ("rsa", "asymmetric_encryption", "RSA algorithm"),
    ("hash_function", "cryptography", "One-way function"),
    ("sha256", "hash_function", "SHA-256 hash"),
    ("digital_signature", "cryptography", "Verifying authenticity"),
    ("certificate", "security", "Digital certificate"),
    ("authentication", "security", "Verifying identity"),
    ("authorization", "security", "Granting access"),
    ("oauth", "authentication", "Open Authorization"),
    ("jwt", "authentication", "JSON Web Token"),
    # Attacks
    ("cyber_attack", "process", "Malicious cyber activity"),
    ("malware", "software", "Malicious software"),
    ("virus", "malware", "Self-replicating malware"),
    ("ransomware", "malware", "Encrypts files for ransom"),
    ("phishing", "cyber_attack", "Social engineering attack"),
    ("ddos", "cyber_attack", "Distributed Denial of Service"),
    ("sql_injection", "cyber_attack", "Injecting SQL code"),
    ("xss", "cyber_attack", "Cross-Site Scripting"),
]


# =============================================================================
# COMPUTER SCIENCE FACTS
# =============================================================================

CS_FACTS = [
    # =========================================================================
    # HARDWARE FACTS
    # =========================================================================
    # CPU facts
    ("cpu", "executes", "instruction"),
    ("cpu", "contains", "alu"),
    ("cpu", "contains", "control_unit"),
    ("cpu", "contains", "register"),
    ("cpu", "contains", "cpu_cache"),
    ("alu", "performs", "arithmetic"),
    ("alu", "performs", "logic_operations"),
    # CPU architectures
    ("x86", "is_a", "cisc"),
    ("arm", "is_a", "risc"),
    ("risc_v", "is_a", "risc"),
    ("cisc", "has_property", "complex_instructions"),
    ("risc", "has_property", "simple_instructions"),
    ("risc", "has_property", "more_registers"),
    # GPU facts
    ("gpu", "optimized_for", "parallel_computation"),
    ("gpu", "contains", "shader_core"),
    ("gpu", "uses", "vram"),
    ("cuda_core", "made_by", "nvidia"),
    ("tensor_core", "accelerates", "matrix_multiplication"),
    ("tensor_core", "used_for", "deep_learning"),
    # Memory hierarchy
    ("l1_cache", "faster_than", "l2_cache"),
    ("l2_cache", "faster_than", "l3_cache"),
    ("l3_cache", "faster_than", "ram"),
    ("ram", "faster_than", "ssd"),
    ("ssd", "faster_than", "hdd"),
    ("sram", "faster_than", "dram"),
    ("dram", "requires", "refresh"),
    ("sram", "does_not_require", "refresh"),
    # Memory types
    ("ddr5", "faster_than", "ddr4"),
    ("ddr4", "faster_than", "ddr3"),
    ("hbm", "has_property", "high_bandwidth"),
    ("hbm", "used_in", "gpu"),
    ("nvme", "faster_than", "sata_ssd"),
    # =========================================================================
    # SOFTWARE FACTS
    # =========================================================================
    # Operating systems
    ("linux", "is_a", "open_source"),
    ("linux_kernel", "written_in", "c"),
    ("linux", "uses", "linux_kernel"),
    ("android", "uses", "linux_kernel"),
    ("macos", "based_on", "bsd"),
    ("ios", "based_on", "macos"),
    ("windows", "uses", "windows_nt"),
    # Programming languages
    ("c", "compiled_to", "machine_code"),
    ("cpp", "superset_of", "c"),
    ("java", "runs_on", "jvm"),
    ("python", "is_a", "dynamically_typed"),
    ("rust", "has_property", "memory_safety"),
    ("rust", "has_property", "no_garbage_collector"),
    ("go", "has_property", "garbage_collection"),
    ("go", "has_property", "concurrency_support"),
    ("haskell", "is_a", "purely_functional"),
    ("javascript", "runs_in", "browser"),
    ("typescript", "superset_of", "javascript"),
    # Language paradigms
    ("python", "supports", "object_oriented"),
    ("python", "supports", "functional"),
    ("python", "supports", "imperative"),
    ("java", "supports", "object_oriented"),
    ("haskell", "supports", "functional"),
    ("prolog", "supports", "logic"),
    # =========================================================================
    # ALGORITHM COMPLEXITY
    # =========================================================================
    # Sorting complexities
    ("quicksort", "has_average_complexity", "O(n log n)"),
    ("quicksort", "has_worst_complexity", "O(n^2)"),
    ("mergesort", "has_complexity", "O(n log n)"),
    ("mergesort", "has_property", "stable"),
    ("heapsort", "has_complexity", "O(n log n)"),
    ("insertion_sort", "has_complexity", "O(n^2)"),
    ("bubble_sort", "has_complexity", "O(n^2)"),
    ("radix_sort", "has_complexity", "O(nk)"),
    # Search complexities
    ("binary_search", "has_complexity", "O(log n)"),
    ("binary_search", "requires", "sorted_array"),
    ("linear_search", "has_complexity", "O(n)"),
    ("hash_lookup", "has_average_complexity", "O(1)"),
    # Graph complexities
    ("bfs", "has_complexity", "O(V+E)"),
    ("dfs", "has_complexity", "O(V+E)"),
    ("dijkstra", "has_complexity", "O(V^2)"),
    ("dijkstra", "with_heap", "O((V+E) log V)"),
    # Data structure operations
    ("array", "access_complexity", "O(1)"),
    ("array", "search_complexity", "O(n)"),
    ("hash_table", "average_access", "O(1)"),
    ("binary_search_tree", "average_access", "O(log n)"),
    ("linked_list", "access_complexity", "O(n)"),
    ("linked_list", "insert_complexity", "O(1)"),
    # =========================================================================
    # COMPLEXITY THEORY FACTS
    # =========================================================================
    ("p", "subset_of", "np"),
    ("np_complete", "subset_of", "np"),
    ("np_complete", "subset_of", "np_hard"),
    ("p_vs_np", "is_a", "unsolved_problem"),
    ("sat", "is_a", "np_complete"),
    ("traveling_salesman", "is_a", "np_hard"),
    ("sorting", "is_a", "p"),
    ("primality_testing", "is_a", "p"),
    # =========================================================================
    # NETWORKING FACTS
    # =========================================================================
    # Protocol stack
    ("tcp", "operates_at", "transport_layer"),
    ("udp", "operates_at", "transport_layer"),
    ("ip", "operates_at", "network_layer_3"),
    ("http", "operates_at", "application_layer"),
    ("ethernet", "operates_at", "data_link_layer"),
    # Protocol properties
    ("tcp", "has_property", "reliable"),
    ("tcp", "has_property", "ordered"),
    ("tcp", "has_property", "connection_oriented"),
    ("udp", "has_property", "unreliable"),
    ("udp", "has_property", "connectionless"),
    ("udp", "has_property", "low_latency"),
    ("https", "uses", "tls"),
    ("https", "has_property", "encrypted"),
    # =========================================================================
    # DATABASE FACTS
    # =========================================================================
    ("relational_database", "uses", "sql"),
    ("relational_database", "has_property", "acid"),
    ("nosql_database", "has_property", "eventual_consistency"),
    ("redis", "stores_in", "memory"),
    ("redis", "has_property", "fast"),
    ("postgresql", "supports", "json"),
    ("postgresql", "has_property", "extensible"),
    ("mongodb", "stores", "documents"),
    ("elasticsearch", "optimized_for", "search"),
    ("vector_database", "optimized_for", "similarity_search"),
    ("faiss", "made_by", "facebook"),
    # =========================================================================
    # DISTRIBUTED SYSTEMS FACTS
    # =========================================================================
    ("cap_theorem", "states", "choose_two_of_cap"),
    ("eventual_consistency", "weaker_than", "strong_consistency"),
    ("kubernetes", "orchestrates", "container"),
    ("docker", "creates", "container"),
    ("kafka", "provides", "event_streaming"),
    ("microservices", "communicates_via", "api"),
    # =========================================================================
    # AI/ML FACTS
    # =========================================================================
    # Learning types
    ("supervised_learning", "requires", "labeled_data"),
    ("unsupervised_learning", "does_not_require", "labeled_data"),
    ("reinforcement_learning", "learns_from", "rewards"),
    # Neural networks
    ("neural_network", "consists_of", "neuron"),
    ("neural_network", "trained_by", "backpropagation"),
    ("backpropagation", "uses", "gradient_descent"),
    ("cnn", "uses", "conv_layer"),
    ("cnn", "good_for", "image_classification"),
    ("rnn", "has_property", "sequential_processing"),
    ("lstm", "solves", "vanishing_gradient"),
    ("transformer", "uses", "self_attention"),
    ("transformer", "has_property", "parallel_processing"),
    # Attention mechanism
    ("attention", "computes", "weighted_sum"),
    ("self_attention", "relates", "positions_in_sequence"),
    ("multi_head_attention", "has_multiple", "attention"),
    # LLMs
    ("llm", "is_a", "transformer"),
    ("llm", "trained_on", "text_data"),
    ("gpt", "uses", "decoder_only"),
    ("bert", "uses", "encoder_only"),
    ("t5", "uses", "encoder_decoder"),
    ("rlhf", "improves", "alignment"),
    ("chain_of_thought", "improves", "reasoning"),
    ("rag", "reduces", "hallucination"),
    # Training
    ("adam", "is_a", "adaptive_optimizer"),
    ("adam", "combines", "momentum"),
    ("adam", "combines", "rmsprop"),
    ("dropout", "prevents", "overfitting"),
    ("batch_norm", "stabilizes", "training"),
    ("layer_norm", "used_in", "transformer"),
    # Vision
    ("resnet", "uses", "skip_connections"),
    ("vit", "applies", "transformer"),
    ("vit", "applied_to", "image_patches"),
    ("stable_diffusion", "generates", "images"),
    ("clip", "connects", "text_and_images"),
    # Quantization
    ("quantization", "reduces", "model_size"),
    ("quantization", "reduces", "inference_time"),
    ("int8", "smaller_than", "fp16"),
    ("fp16", "smaller_than", "fp32"),
    ("mixed_precision", "uses", "fp16"),
    ("mixed_precision", "uses", "fp32"),
    # Frameworks
    ("pytorch", "uses", "dynamic_computation_graph"),
    ("tensorflow", "uses", "static_computation_graph"),
    ("jax", "has_property", "functional"),
    ("jax", "supports", "automatic_differentiation"),
    # =========================================================================
    # SECURITY FACTS
    # =========================================================================
    ("aes", "is_a", "block_cipher"),
    ("aes", "key_sizes", "128_192_256_bits"),
    ("rsa", "based_on", "prime_factorization"),
    ("sha256", "produces", "256_bit_hash"),
    ("https", "encrypts", "web_traffic"),
    ("tls", "replaced", "ssl"),
    ("oauth", "provides", "delegated_authorization"),
    ("jwt", "contains", "claims"),
    # =========================================================================
    # FAMOUS RESULTS
    # =========================================================================
    ("turing_machine", "proposed_by", "alan_turing"),
    ("turing_machine", "equivalent_to", "lambda_calculus"),
    ("halting_problem", "is_a", "undecidable"),
    ("halting_problem", "proven_by", "alan_turing"),
    ("church_turing_thesis", "states", "computable_equals_turing_computable"),
    ("moores_law", "states", "transistors_double_every_two_years"),
    ("moores_law", "proposed_by", "gordon_moore"),
    # =========================================================================
    # COMPANIES AND PRODUCTS
    # =========================================================================
    # GPU companies
    ("nvidia", "makes", "gpu"),
    ("nvidia", "created", "cuda"),
    ("nvidia", "makes", "tensor_core"),
    ("amd", "makes", "gpu"),
    ("amd", "makes", "cpu"),
    ("intel", "makes", "cpu"),
    ("intel", "makes", "gpu"),
    ("apple", "makes", "arm"),
    ("arm_holdings", "designs", "arm"),
    # AI companies
    ("openai", "created", "gpt"),
    ("openai", "created", "dall_e"),
    ("anthropic", "created", "claude"),
    ("google", "created", "bert"),
    ("google", "created", "transformer"),
    ("google", "created", "tensorflow"),
    ("meta", "created", "pytorch"),
    ("meta", "created", "llama"),
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    # Training relationships
    ("more_data", "improves", "model_performance"),
    ("more_parameters", "increases", "model_capacity"),
    ("larger_batch_size", "improves", "training_stability"),
    ("higher_learning_rate", "speeds_up", "training"),
    ("higher_learning_rate", "risks", "divergence"),
    # Hardware relationships
    ("more_cuda_cores", "increases", "gpu_throughput"),
    ("more_memory", "enables", "larger_models"),
    ("faster_interconnect", "improves", "multi_gpu_training"),
    ("tensor_cores", "accelerate", "matrix_operations"),
    # Scaling laws
    ("compute", "scales_with", "model_performance"),
    ("data", "scales_with", "model_performance"),
    ("parameters", "scale_with", "model_performance"),
]


# =============================================================================
# LOADER FUNCTION
# =============================================================================


def load_cs_into_core(core, agent_id: str = None) -> int:
    """
    Load all computer science knowledge into a Dorian Core.
    """
    if agent_id is None:
        cs_agent = core.register_agent(
            "cs_loader", domain="computer_science", can_verify=True
        )
        agent_id = cs_agent.agent_id

    count = 0

    print(f"  Loading {len(CS_CATEGORIES)} CS categories...")
    for name, parent, description in CS_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="computer_science",
                    level=parent_level + 1,
                )
            )

        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="cs_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    print(f"  Loading {len(CS_FACTS)} CS facts...")
    for s, p, o in CS_FACTS:
        result = core.write(
            s, p, o, agent_id, source="cs_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} CS facts loaded")
    return count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN COMPUTER SCIENCE DOMAIN")
    print("═" * 60)

    print(f"\nCategories: {len(CS_CATEGORIES)}")
    print(f"Facts: {len(CS_FACTS)}")

    # Count by area
    areas = {}
    for name, parent, desc in CS_CATEGORIES:
        if "gpu" in name or "cuda" in name or "tensor_core" in name:
            areas["GPU"] = areas.get("GPU", 0) + 1
        elif "cpu" in name or "alu" in name or "cache" in name:
            areas["CPU"] = areas.get("CPU", 0) + 1
        elif "memory" in name or "ram" in name or "ddr" in name:
            areas["Memory"] = areas.get("Memory", 0) + 1
        elif "neural" in name or "learning" in name or "ml" in name or "ai" in name:
            areas["AI/ML"] = areas.get("AI/ML", 0) + 1
        elif "network" in name or "protocol" in name or "tcp" in name:
            areas["Networking"] = areas.get("Networking", 0) + 1
        elif "database" in name or "sql" in name:
            areas["Databases"] = areas.get("Databases", 0) + 1
        elif "algorithm" in name or "sort" in name or "search" in name:
            areas["Algorithms"] = areas.get("Algorithms", 0) + 1

    print("\nCategories by area:")
    for area, count in sorted(areas.items(), key=lambda x: -x[1]):
        print(f"  {area}: {count}")

    print("\nSample categories:")
    for name, parent, desc in CS_CATEGORIES[:15]:
        print(f"  {name} <- {parent}")

    print("\nSample facts:")
    for s, p, o in CS_FACTS[:15]:
        print(f"  {s} {p} {o}")
