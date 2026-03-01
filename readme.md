# Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation.

> **Note**: This is a reproduction of the IJCAI 2022 paper by Peng et al. The original repository is [pengwei-iie/GLHG](https://github.com/pengwei-iie/GLHG). This fork contains a full reimplementation including the Hierarchical Graph Reasoner (`models/hierarchical_graph.py`) and GLHG inputter (`inputters/glhg.py`).

## Documentation

- [`docs/GLHG_GUIDE.md`](docs/GLHG_GUIDE.md) — Step-by-step 실행 가이드 (prepare → train → infer) 및 재현 결과
- [`docs/GLHG_IMPLEMENTATION.md`](docs/GLHG_IMPLEMENTATION.md) — 아키텍처 상세 설명 (Multi-source Encoder, Hierarchical Graph Reasoner, Global-guide Decoder)

---

This is the repository of our IJCAI 2022 paper Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation.

If your want to make a **human evaluation** with GLHG. **The results are available in generation-glhg.json**. And **automatic evaluation** can be use in the following.

![image](https://user-images.githubusercontent.com/30322673/233757464-28a32f63-fb5b-4bd9-81c6-ea4d690ff18a.png)


If you use this baseline, we would appreciate you citing our work:
    
```
@inproceedings{DBLP:conf/ijcai/00080XXSL22,
  author    = {Wei Peng and
               Yue Hu and
               Luxi Xing and
               Yuqiang Xie and
               Yajing Sun and
               Yunpeng Li},
  editor    = {Luc De Raedt},
  title     = {Control Globally, Understand Locally: {A} Global-to-Local Hierarchical
               Graph Network for Emotional Support Conversation},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  pages     = {4324--4330},
  publisher = {ijcai.org},
  year      = {2022},
  url       = {https://doi.org/10.24963/ijcai.2022/600},
  doi       = {10.24963/ijcai.2022/600},
  timestamp = {Wed, 27 Jul 2022 16:43:00 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/00080XXSL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
