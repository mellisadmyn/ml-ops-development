# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

Proyek ini merupakan bagian dari submission akhir pada kelas **Machine Learning Operations (MLOps)** di Dicoding. Tujuan utama dari proyek ini adalah untuk membuatan machine learning pipeline sederhana menggunakan TensorFlow Extended (TFX) menggunakan komputasi cloud.

## 🎯 Objectives

1. **Membangun Machine Learning Pipeline End-to-End dengan TFX**
   
    Mengimplementasikan ML pipeline menggunakan TensorFlow Extended (TFX) dengan komponen-komponen utama seperti `ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Resolver, Evaluator, dan Pusher` untuk memastikan alur kerja yang terstandarisasi. Pipeline dijalankan menggunakan orchestrator Apache Beam.

2. **Menerapkan Sistem Machine Learning di Platform Cloud**

    Men-deploy model machine learning ke layanan cloud (Railway) agar dapat diakses sebagai web service yang siap digunakan oleh user.

3. **Melakukan Monitoring Sistem Menggunakan Prometheus**

    Mengintegrasikan Prometheus untuk memantau performa dan kestabilan sistem machine learning yang telah dideploy di cloud.


## 📑 Methodology

### 1️⃣ Pipeline Configuration
- **Inisialisasi Parameter**: Mendefinisikan nama pipeline, direktori input data, direktori untuk menyimpan model hasil pelatihan, dan path untuk menyimpan metadata.
- **Pembuatan Struktur Folder**: Membuat folder `modules` berisi file `transform.py`, `trainer.py`, `tuner.py`, dan `components.py` yang dibutuhkan untuk menyusun pipeline modular.

### 2️⃣ TFX Components Initialization
- **Modularisasi Komponen**: Komponen-komponen utama pipeline seperti `ExampleGen`, `StatisticsGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Tuner`, `Trainer`, `Resolver`, `Evaluator`, dan `Pusher` didefinisikan di dalam `components.py`.
- **Import Komponen**: Komponen-komponen ini kemudian diimpor dan digunakan untuk membangun pipeline secara lengkap.

### 3️⃣ Pipeline Orchestration with Apache Beam
- **Membangun Pipeline TFX**: Menggunakan `pipeline.Pipeline()` untuk menyusun seluruh alur komponen menjadi satu kesatuan workflow.
- **Runner Configuration**: Menggunakan `BeamDagRunner` untuk menjalankan pipeline secara lokal menggunakan Apache Beam sebagai orchestrator.

### 4️⃣ Model Exporting
- **Serving Model**: Model hasil training disimpan di direktori `output/serving_model` untuk kemudian digunakan dalam proses deployment.

### 8️⃣ Model Serving Test  
- **Pengujian Model via Railway**: Model yang telah di-deploy diakses melalui endpoint REST API menggunakan platform Railway.


## 📊 Submission Review

<img width="924" alt="Score Submission" src="https://github.com/user-attachments/assets/5cb31b5d-c6f4-40e7-9314-a283e34cc45b" />


## 📌 Conclusion
Proyek ini telah berhasil membangun machine learning pipeline secara modular menggunakan TensorFlow Extended (TFX) dan dijalankan melalui Apache Beam. Pipeline mencakup proses mulai dari ingest data, validasi, transformasi fitur, training, evaluasi, hingga deployment model. Model hasil pelatihan kemudian di-deploy menggunakan TensorFlow Serving dan diuji melalui endpoint REST API di Railway. Proyek ini menunjukkan alur MLOps end-to-end yang dapat digunakan untuk mendukung prediksi risiko attrition karyawan secara berkelanjutan.

**Author:** Mellisa  
**Date:** 17-12-2024  
**Tools:** Python, TensorFlow, TensorFlow Extended (TFX), TFX Components, TensorFlow Serving, Railway