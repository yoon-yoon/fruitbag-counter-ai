# 🍐 FruitBag Counter AI

A deep learning-based system for accurately counting fruit bags in orchard environments using object detection, object tracking, and line-crossing algorithms. Developed as part of the 2025 AI Convergence Project at Chungnam National University.

---

## 📌 Project Overview

This project aims to automate the labor-intensive task of counting fruit bags in orchards. By leveraging **YOLO-based object detection**, **BoT-SORT tracking**, and a **custom line-crossing counting algorithm**, the system achieves high accuracy and robustness even in complex field environments.

- 🔍 Object detection using YOLOv11
- 🧠 Object tracking with BoT-SORT
- 🚦 Line-crossing algorithm to eliminate duplicate counts
- 🌐 Web-based interface for easy video analysis and visualization

---

## 📊 Performance Summary

| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| Validation Set | 99.1% | 98.2% | 98.8% |
| Test Set       | 93.8% | 89.8% | 87.7% |

| Method | Counting Accuracy (%) | MAE (count) |
|--------|------------------------|-------------|
| YOLO + Tracking | 26.3% | 21.0 |
| YOLO + Tracking + Line-Crossing | **91.2%** | **2.5** |

---

## 🧩 System Architecture

![Framework](./images/framework.png)  
*Figure 1. Overall Framework of the Proposed System*

![Web Interface](./images/web_interface.png)  
*Figure 2. Web-based Interface Configuration*

---

## 🚀 How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/yoon-yoon/fruitbag-counter-ai.git
   cd fruitbag-counter-ai
  ```

2. Set up your environment

   ```bash
   conda create -n fruitbag python=3.10
   conda activate fruitbag
   pip install -r ./streamlit/requirements.txt
   ```

3. Launch the web interface

   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
fruitbag-counter-ai/
├── data/                  # Preprocessed dataset (YOLO format)
├── models/                # Trained weights
├── tracking/              # BoT-SORT tracking module
├── line_crossing/         # Counting algorithm
├── app.py                 # Streamlit interface
├── detect_and_count.py    # Core detection + tracking + counting script
└── README.md
```

---

## 📚 Acknowledgements

* AI Hub Dataset (Orchard Object Detection)
* YOLOv11 and BoT-SORT open-source implementations
* Supported by Chungnam National University, AI Convergence Graduate Program

---

## 🧑‍💻 Author

* 🧾 Name: Kyungyoon Yoon (윤경윤)
* 📧 Email: [kyungji123@naver.com](mailto:kyungji123@naver.com)
* 📅 Project Duration: Mar 5 – Jun 18, 2025
* 📍 Affiliation: Department of AI Convergence, Chungnam National University
