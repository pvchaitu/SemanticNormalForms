# ---

**SDNF Demo: Semantic Data Normal Forms and SRS Pipeline**

**Version: 0.1.0-alpha**

**Status: Functional Prototype / Research Artifact**

This repository provides a hands-on implementation of the Semantic Data Normal Forms (SDNF) framework and the Semantic Relational Schema (SRS). It addresses the integrity, stability, and explainability challenges of AI-native databases by moving from structural constraints to semantic constraints.

## ---

**System Architecture and Workflow**

The following flowchart illustrates the end-to-end pipeline, highlighting where specific files are utilized and how they implement the theoretical novelties of the SDNF framework.

### **Pipeline Walkthrough**

1. **Data Ingestion (schema\_ingest.py)**:  
   * Validates raw payloads against ISO 20022 and PCI-DSS standards.  
   * Derives a draft schema, ensuring Entity Embedding Normal Form (EENF) readiness.  
2. **Semantic Embedding (embedding\_utils.py)**:  
   * **Novelty**: Implements Multi-level Embeddings (Fine-grained and Abstract levels).  
   * **Privacy**: Injects Gaussian Differential Privacy (DP) noise into sensitive fields like PAN to prevent vector reconstruction.  
3. **AANF Enforcement (semantic\_merge.py)**:  
   * **Novelty**: Uses Evidence Sets (ECNF) to merge aliases based on threshold m\_min.  
4. **CMNF Contextualization (cmnf.py)**:  
   * **Novelty**: Learns Context Modulation Normal Form (CMNF) projections.  
   * **Orthogonality Check**: Enforces a penalty to ensure different contexts remain orthogonal.  
5. **Persistence and Audit (db\_utils.py)**:  
   * Stores merge decisions and projection matrices in a Lineage Table for explainability.  
6. **Visualization (visualize.py)**:  
   * Generates t-SNE plots comparing original vs. context-projected embeddings.

## ---

**Quick Start**

### **1\. Setup Virtual Environment**

It is highly recommended to use a virtual environment to manage the sentence-transformers and numpy dependencies.

**On Unix/macOS:**

Bash

python3 \-m venv venv  
source venv/bin/activate

**On Windows:**

Bash

python \-m venv venv  
.\\venv\\Scripts\\activate

### **2\. Install Requirements**

Once the environment is active, install the necessary research libraries:

Bash

pip install \-r requirements.txt

### **3\. Run Demo**

Execute the main research script to view the SRS evolution and SDNF compliance table:

Bash

python demo.py

*Note: The demo will log detailed debug information, including EENF variance and CMNF orthogonality checks.*

## ---

**Contribution Section**

We welcome contributions that extend the SDNF theory or improve implementation efficiency.

* **D-BNF Implementation**: Help implement Drift-Bounded Normal Form logic to detect semantic shift.  
* **Optimization**: We welcome pull requests for approximate orthogonality using Locality Sensitive Hashing (LSH).  
* **Standards Mapping**: Help map more JSON attributes to ISO 20022 or FHIR standards.

## ---

**Standards Reference**

* **ISO 20022**: Used for naming conventions in INAmex.json.  
* **PCI-DSS**: Principles for PII handling and DP noise applied in embedding\_utils.py.  
* **IEEE SDNF Framework**: Companion code to "Semantic Data Normal Forms: A Framework for Embedding-Native Schema Integrity".

## ---

**Troubleshooting and Debugging**

* **File Not Found**: Ensure INAmex.json and PPVisa.json are in the root or data/ folder.  
* **ImportError**: If sentence-transformers is missing, ensure your virtual environment is active and requirements are installed.  
* **Visualization Failures**: Ensure matplotlib and scikit-learn are installed to generate t-SNE plots.

---

