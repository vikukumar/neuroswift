# Contributing to NeuroSwift

Welcome to the NeuroSwift community! We are building the most powerful and efficient CPU-first deep learning architecture in the world. Whether you are an AI researcher, a data scientist, or an MLOps engineer, your contributions are invaluable.

---

## 🌟 How You Can Help

- **Architecture Research**: Propose new linear token mixing layers or improved gating for Sparse MoE.
- **Data Science**: Add new schema mappers for specialized domain data (e.g., medical, financial).
- **Performance Tuning**: Add new optimizations for specific CPU instruction sets (AVX-512, AMX).
- **Ecosystem**: Build new examples or integrate with other tools (e.g., LangChain, Hugging Face).

## 🚀 Getting Started

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/neuroswift.git
   cd neuroswift
   ```
2. **Environment Setup**:
   ```bash
   setup_env.bat  # Windows
   # or
   bash setup_env.sh # Linux/macOS
   ```
3. **Run Benchmarks**:
   Ensure your environment is performant:
   ```bash
   python -m neuroswift benchmark
   ```

## 🛠 Development Guidelines

- **Code Style**: We use `black` for formatting and `isort` for import sorting.
- **Testing**: Add unit tests in `tests/` for any new layer or logic.
- **Documentation**: If you add a feature, update the `README.md` and `docs/`.

## 🧪 Submission Process

1. Create a descriptive branch: `git checkout -b feat/ultra-fast-scan`.
2. Commit your changes with clear messages.
3. Push and open a **Pull Request**.
4. Our "God-Level" maintainers will review and merge!

---
**Thank you for helping us define the future of intelligence!**
