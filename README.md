# Dynamic Optimization Beam Search (DOBS)

**Abstract:**  With the increasing number of applications relying on Large Language Models (LLMs), many LLM manufacturers have incorporated automatic prompt engineering into their GUIs to improve the precision of user input and achieve more accurate responses. However, due to the diversity of user requirements faced by LLMS, using automatic prompt optimization strategies by default in various scenarios may misinterpret user intentions and lead to unintentional prompt negative optimization, resulting in erroneous LLM outputs. In this paper, the dynamic optimization beam search method is proposed to restore this scenario, which simulates most automatic prompt engineering as a method that can ensure the dynamic stability of the semantics of attack samples to explore the impact of the widespread application of this kind of strategy on LLM. This study achieves good attack results on various multi-category open-source and closed-source models, which provides valuable insights for the design and robustness of automatic prompt projects.

![The core of dynamic optimization beam search.](Workflow.png)

Please install ollama first. Linux server samples: ![Ollama](https://ollama.com/download/linux)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull [target models] 
```

Here, our target models in experiments can be found in: ![Ollama Model List](https://ollama.com/search)

