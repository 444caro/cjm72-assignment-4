# Latent Semantic Analysis (LSA) Search Engine Web Application

This project is a simple web-based search engine that uses **Latent Semantic Analysis (LSA)** to perform semantic document retrieval. A user can enter a query, and the application returns the most relevant documents based on **cosine similarity in reduced latent space**. The results are displayed along with a bar chart visualizing similarity scores.

## Features

- Front-end web UI with a search form and dynamic results display.
- Back-end handles LSA-based search and returns relevant documents.
- Cosine similarity chart visualizes relevance scores.
- No page reloads — uses JavaScript `fetch()` to call server.

---

## ✨ Demo Overview

1. **User enters a query** in the input field.
2. On submission, a POST request is sent to `/search` with the query.
3. The back-end:
   - Vectorizes the query using the LSA-transformed term-document matrix.
   - Computes cosine similarity between the query vector and document vectors.
   - Returns the top matching documents, their indices, and similarity scores.
4. The front-end:
   - Displays the results in a readable format.
   - Renders a bar chart showing the similarity of each document.

---
