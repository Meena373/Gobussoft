# Globussoft Data Science Assignment

## Project Structure

├── amazon_laptop_scraper.ipynb        # Task 1 – Amazon Laptop Scraper  
├── face_auth_training.py / .ipynb     # Task 2 – Face Authentication Training  
├── face_auth_testing.py / .ipynb      # Task 2 – Face Authentication Inference / Testing  
├── requirements.txt                   # Python dependencies  
├── README.md                          # Project documentation  
└── sample_images/                     # Optional sample images for testing  

---

## Task 1 – Amazon Laptop Scraper

### Objective
Scrape laptop product data from Amazon India and store the results in a timestamped CSV file.

### Extracted Fields
- Image URL
- Product Title
- Rating
- Price
- Ad / Organic Result

### Features
- Multi‑page scraping
- Browser‑like request headers
- Logging support
- Timestamped CSV export

### Run
```bash
jupyter notebook amazon_laptop_scraper.ipynb
```

Output:
A CSV file will be generated automatically with a timestamp.

---

## Task 2 – Face Authentication (Face Verification)

### Objective
Build a Python + FastAPI‑ready face verification system.

### Workflow
1. Accept two face images
2. Detect face(s)
3. Extract embeddings
4. Compute similarity score
5. Return:
   - Verification result (same person / different person)
   - Similarity score
   - Bounding boxes

### Model Stack
- FaceNet / InceptionResnetV1
- MTCNN Face Detection
- PyTorch
- OpenCV
- FastAPI compatible implementation

### Training
Run the training notebook/script:

```bash
jupyter notebook face_auth_training.ipynb
```

Saved model:
```text
face_auth_model.pth
```

### Testing / Inference

```bash
jupyter notebook face_auth_testing.ipynb
```

Use:
```python
load_model()
predict(image1, image2)
```

---

## Installation

Create environment:

```bash
pip install -r requirements.txt
```

---

## Technologies Used
- Python
- PyTorch
- FastAPI
- OpenCV
- Facenet‑Pytorch
- BeautifulSoup
- Requests
- Pandas
- NumPy

