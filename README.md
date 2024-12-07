# ChatBot_RAG

![Uploading ảnh.png…]()


Chat bot ứng dụng RAG để học có thể tận dụng được nguồn dữ liệu private bên ngoài 

Công nghệ sử dụng 
* Langchain: tương tác với LLM
* ElasticSearch: triển khai các thuật toán search
* Gradio: triển khai chatbot
* Ollama: host LLM trên local

### Một số yêu cầu khi chạy

* ChatBot được chạy trên local vì vậy, đảm bảo máy có GPU
* Pull model LLM trên ollama về trước khi run
* Có thể bổ sung thêm data
* Có thể sử dụng các mô hình embeddings, llm ollama khác trong code
* Có thể sử dụng các loại Elastic Search khác như: Hybrid, ...

### Cách chạy

``` sh run.sh ```

hoặc

```python gradio_elastic_search.py -public 0 -domain 0 -model_ollama 4```


