# Alusive-ML-Solutions-Final-Version

## Alusive Africa: Machine Learning Solutions

### 1. Grant Allocator
The **Grant Allocation Model** is a machine learning-based solution designed to optimize the distribution of educational grants at Alusive Africa. The model predicts the optimal grant amount for each applicant based on their profile, ensuring an equitable and impactful allocation of funds.

![Grant Allocator Page](<Grant Allocator.png>)

#### Key Features:
- **Classification-based Prediction**: Estimates the appropriate grant category (low, medium, high) based on applicant profiles.
- **Fair and Impact-driven Distribution**: Ensures the total allocation remains within the budget constraint (USD 30,000 for 45 students).
- **Feature Engineering**: Considers multiple factors, such as financial standing, academic background, and household support.
- **Optimization Strategies**: Balances the need to support as many students as possible while maximizing the impact of the grants.

### 2. Document Validator
The **Document Validator** leverages deep learning techniques to automate the verification of submitted documents, ensuring authenticity and compliance with grant application requirements.

![Document Validator Page](<Document Validator.png>)

#### Key Features:
- **RASNET Model for Classification**: Differentiates between signed and unsigned documents.
- **Image Processing & Augmentation**: Enhances model robustness against variations in document quality.
- **Class Balancing & Optimization**: Adjusts for dataset imbalances to improve prediction accuracy.
- **Automated Verification**: Speeds up the validation process, reducing manual effort and minimizing errors.

### 3. Alusive Africa Chatbot
The **Alusive Africa Chatbot** The Alusive Africa Chatbot API is designed to provide instant, AI-powered responses to frequently asked questions about Alusive Africa. Built using FastAPI, the API leverages a pretrained Sentence Transformer model (all-MiniLM-L6-v2) to understand and match user queries with a curated knowledge base of FAQs.

![Chatbot Page](<Chatbot.png>)

#### Key Features:
- **Grant Support Information:**: Understands and responds to user queries in a conversational manner.
- **Internship Program Insights:**: Guides applicants through the grant application process and eligibility criteria.
- **Student Venture Support:**: Provides instructions and validation feedback for document uploads.

---
These ML solutions collectively improve efficiency, transparency, and accessibility for grant allocation and document validation, reinforcing Alusive Africa's mission to support students in accessing world-class education.

### Running the Application

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

Download a zip file of the website [here](https://drive.google.com/file/d/1MxfSTtgB-5QnG692B5x2mJS1ZAQyb2Tu/view?usp=share_link)

### Getting Started

Unzip the alusive_website folder

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

### alusive_fastapi_server


To build the project using docker, 
use

1. `docker build -t fastapi-app .`
2. `docker run -p 80:80 fastapi-app` 

# [Video Demo](https://youtu.be/A1dE2QeORjk)