FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit plotly
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY SPEC.md .
RUN mkdir -p artifacts data
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
