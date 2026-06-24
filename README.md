📌 Federated Learning for Lung Cancer Classification (Flower + PyTorch)
# 🫁 Federated Learning for Lung Cancer Classification

This project implements a **Federated Learning (FL) system** for lung cancer image classification using **Flower Framework** and **PyTorch**.

The system simulates a real-world healthcare scenario where multiple hospitals collaboratively train a deep learning model without sharing patient data.

---

## 📊 Problem Statement

Medical data is highly sensitive and cannot be shared directly between hospitals.

This project solves this problem using **Federated Learning**, where:
- Each hospital trains locally
- Only model weights are shared
- Data never leaves the hospital

---

## 🏥 Dataset

- Type: Lung CT / X-ray images
- Classes:
  - Benign
  - Malignant
  - Normal

### Data Distribution (Non-IID)


client1/ → benign, malignant, normal
client2/ → benign, malignant, normal
client3/ → benign, malignant, normal
client4/ → benign, malignant, normal


Each client contains ~800 images.

---

## ⚙️ Tech Stack

- Python 🐍
- PyTorch 🔥
- Flower (FL Framework) 🌸
- TorchVision
- CNN / ResNet18

---

## 🧠 Model Architecture

We use **ResNet18** modified for 3-class classification:


Input Image (224×224)
↓
ResNet18
↓
Fully Connected Layer (3 outputs)
↓
Benign / Malignant / Normal


---

## 🏗️ System Architecture

            🖥️ Server
                |
---------------------------------
|        |        |            |

Client 1 Client 2 Client 3 Client 4
(Hospital A - D)


---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
2️⃣ Start Server (Run on main machine)
python server.py

Server IP example:

0.0.0.0:8080
3️⃣ Start Clients (Run on each machine)

On Client 1:

python client1.py

On Client 2:

python client2.py

On Client 3:

python client3.py

On Client 4:

python client4.py
📈 Output Example
Round 1 → Accuracy: 78%
Round 2 → Accuracy: 84%
Round 3 → Accuracy: 88%
Round 4 → Accuracy: 91%
🔬 Research Contributions
Federated Learning applied to medical imaging
Privacy-preserving AI for healthcare
Multi-client distributed training (4 hospitals)
Non-IID data simulation
Real-world FL pipeline using Flower
📊 Future Improvements
Add FedProx / FedAvg comparison
Add Differential Privacy
Add Secure Aggregation
Extend to 3D CT scan datasets
Improve CNN backbone (EfficientNet / DenseNet)
👨‍💻 Author

Jakaria , Nadira
BSc Software Engineering (Data Science)
Daffodil International University

📜 License

This project is for academic and research purposes.
