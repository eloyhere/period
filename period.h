#include <complex>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <string>
#include <sstream>
#include <memory>
#include <functional>

namespace neuron
{

// ---------- 辅助函数 ----------
inline double fix(double value, double epsilon = 1e-12)
{
    return (std::abs(value) < epsilon) ? 0.0 : value;
}

inline std::complex<double> fix(const std::complex<double>& z, double epsilon = 1e-12)
{
    return {fix(z.real(), epsilon), fix(z.imag(), epsilon)};
}

// ===================== Neuron =====================
class Neuron
{
public:
    std::complex<double> z;
    double omega;
    double offset;

    Neuron(double amplitude = 1.0, double omega = 1.0, double phi = 0.0, double offset = 0.0)
        : z(std::polar(amplitude, phi)), omega(omega), offset(offset) {}

    explicit Neuron(const std::complex<double>& state,
                    double omega = 1.0, double offset = 0.0)
        : z(state), omega(omega), offset(offset) {}

    double amplitude() const { return std::abs(z); }
    double phi() const { return std::arg(z); }
    double real() const { return z.real(); }
    double imag() const { return z.imag(); }

    void setAmplitude(double a) { z = std::polar(a, phi()); }
    void setPhi(double p) { z = std::polar(amplitude(), p); }

    Neuron bind(const Neuron& other) const
    {
        return Neuron(z * other.z, omega, offset + other.offset);
    }

    Neuron unbind(const Neuron& other) const
    {
        if (other.amplitude() == 0.0)
            throw std::domain_error("Unbind by zero");
        return Neuron(z / other.z, omega, offset - other.offset);
    }

    double sine(double x) const
    {
        return fix(amplitude() * std::sin(omega * x + phi()) + offset);
    }

    double cosine(double x) const
    {
        return fix(amplitude() * std::cos(omega * x + phi()) + offset);
    }

    std::complex<double> gradient() const
    {
        return fix(std::complex<double>(-imag(), real()));
    }

    double similarity(const Neuron& other) const
    {
        if (amplitude() == 0.0 || other.amplitude() == 0.0) return 0.0;
        return (std::conj(z) * other.z).real() / (amplitude() * other.amplitude());
    }

    double distance(const Neuron& other) const
    {
        if (amplitude() == 0.0 || other.amplitude() == 0.0) return 0.0;
        return (std::conj(z) * other.z).imag() / (amplitude() * other.amplitude());
    }

    void update(double lr, const std::complex<double>& grad_z)
    {
        z -= lr * grad_z;
    }

    double loss(const Neuron& target) const
    {
        std::complex<double> diff = z - target.z;
        return fix(std::norm(diff));
    }

    // 运算符
    Neuron operator+(const Neuron& o) const { return Neuron(z + o.z, omega, offset + o.offset); }
    Neuron operator-(const Neuron& o) const { return Neuron(z - o.z, omega, offset - o.offset); }
    Neuron operator*(const Neuron& o) const { return bind(o); }
    Neuron operator/(const Neuron& o) const { return unbind(o); }
    Neuron operator*(double k) const { return Neuron(z * k, omega, offset * k); }
    Neuron operator/(double k) const { return Neuron(z / k, omega, offset / k); }
    friend Neuron operator*(double k, const Neuron& n) { return n * k; }
    Neuron operator-() const { return Neuron(-z, omega, -offset); }
    Neuron operator~() const { return Neuron(std::conj(z), omega, offset); }
    bool operator==(const Neuron& o) const { return z == o.z && omega == o.omega && offset == o.offset; }
    bool operator!=(const Neuron& o) const { return !(*this == o); }
    bool operator<(const Neuron& o) const { return amplitude() < o.amplitude(); }
    bool operator>(const Neuron& o) const { return amplitude() > o.amplitude(); }
    bool operator<=(const Neuron& o) const { return amplitude() <= o.amplitude(); }
    bool operator>=(const Neuron& o) const { return amplitude() >= o.amplitude(); }

    friend std::ostream& operator<<(std::ostream& os, const Neuron& n)
    {
        os << "amplitude: " << n.amplitude() << ", phi: " << n.phi()
           << ", omega: " << n.omega << ", offset" << n.offset << "]";
        return os;
    }

    Neuron conjugate() const { return Neuron(std::conj(z), omega, offset); }
    Neuron inverse() const { return Neuron(1.0 / z, omega, offset); }
    Neuron normalize() const { double a = amplitude(); return a == 0.0 ? *this : Neuron(z / a, omega, offset); }
    double magnitude() const { return amplitude(); }
    double phase() const { return phi(); }

    Neuron sigmoid() const { double a = amplitude(); double s = 1.0/(1.0+std::exp(-a)); return Neuron(s, omega, phi(), offset); }
    Neuron tanh() const { double a = amplitude(); double t = std::tanh(a); return Neuron(t, omega, phi(), offset); }
    Neuron relu() const { double a = amplitude(); return Neuron(a > 0.0 ? a : 0.0, omega, phi(), offset); }
    Neuron exp() const { return Neuron(std::exp(z), omega, offset); }
    Neuron log() const { return Neuron(std::log(z), omega, offset); }
    Neuron pow(double exponent) const { return Neuron(std::pow(z, exponent), omega, offset); }

    static Neuron random(double omega = 1.0, double offset = 0.0)
    {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> ampDist(0.0, 1.0);
        static std::uniform_real_distribution<double> phiDist(0.0, 2.0 * M_PI);
        return Neuron(ampDist(rng), omega, phiDist(rng), offset);
    }
};

// ===================== Layer =====================
class Layer
{
public:
    std::vector<Neuron> neurons;

    Layer() = default;

    Layer(size_t size, double default_omega = 1.0)
    {
        neurons.reserve(size);
        for (size_t i = 0; i < size; ++i)
            neurons.emplace_back(1.0, default_omega, 0.0, 0.0);
    }

    explicit Layer(const std::vector<Neuron>& n) : neurons(n) {}

    size_t size() const { return neurons.size(); }

    Neuron& operator[](size_t i) { return neurons[i]; }
    const Neuron& operator[](size_t i) const { return neurons[i]; }

    Layer bind(const Layer& other) const
    {
        if (size() != other.size()) throw std::domain_error("Layer size mismatch for bind");
        Layer result;
        result.neurons.reserve(size());
        for (size_t i = 0; i < size(); ++i)
            result.neurons.push_back(neurons[i].bind(other.neurons[i]));
        return result;
    }

    Layer unbind(const Layer& other) const
    {
        if (size() != other.size()) throw std::domain_error("Layer size mismatch for unbind");
        Layer result;
        result.neurons.reserve(size());
        for (size_t i = 0; i < size(); ++i)
            result.neurons.push_back(neurons[i].unbind(other.neurons[i]));
        return result;
    }

    Layer operator+(const Layer& o) const {
        if (size() != o.size()) throw std::domain_error("Size mismatch");
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] + o.neurons[i]);
        return res;
    }

    Layer operator-(const Layer& o) const {
        if (size() != o.size()) throw std::domain_error("Size mismatch");
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] - o.neurons[i]);
        return res;
    }

    Layer operator*(const Layer& o) const { return bind(o); }
    Layer operator/(const Layer& o) const { return unbind(o); }

    Layer operator*(double k) const {
        Layer res;
        for (auto& n : neurons) res.neurons.push_back(n * k);
        return res;
    }

    Layer operator/(double k) const {
        Layer res;
        for (auto& n : neurons) res.neurons.push_back(n / k);
        return res;
    }

    friend Layer operator*(double k, const Layer& l) { return l * k; }

    Neuron weighted_sum(const std::vector<double>& weights) const
    {
        if (weights.size() != size()) throw std::domain_error("Weight size mismatch");
        std::complex<double> sum(0.0, 0.0);
        double omega = neurons.empty() ? 1.0 : neurons[0].omega;
        double offset = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            sum += weights[i] * neurons[i].z;
            offset += weights[i] * neurons[i].offset;
        }
        double total_w = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (total_w == 0.0) return Neuron();
        return Neuron(sum / total_w, omega, offset / total_w);
    }

    double similarity(const Layer& other) const
    {
        if (size() != other.size() || size() == 0) return 0.0;
        double sum = 0.0, wsum = 0.0;
        for (size_t i = 0; i < size(); ++i)
        {
            double ampA = neurons[i].amplitude(), ampB = other.neurons[i].amplitude();
            if (ampA == 0.0 || ampB == 0.0) continue;
            double cos_sim = neurons[i].similarity(other.neurons[i]);
            double w = ampA * ampB;
            sum += cos_sim * w;
            wsum += w;
        }
        return wsum == 0.0 ? 0.0 : sum / wsum;
    }

    double distance(const Layer& other) const
    {
        if (size() != other.size() || size() == 0) return 0.0;
        double sum = 0.0, wsum = 0.0;
        for (size_t i = 0; i < size(); ++i)
        {
            double ampA = neurons[i].amplitude(), ampB = other.neurons[i].amplitude();
            if (ampA == 0.0 || ampB == 0.0) continue;
            double sin_dist = neurons[i].distance(other.neurons[i]);
            double w = ampA * ampB;
            sum += sin_dist * w;
            wsum += w;
        }
        return wsum == 0.0 ? 0.0 : sum / wsum;
    }

    double loss(const Layer& other) const
    {
        if (size() != other.size()) return 1e9;
        double total = 0.0;
        for (size_t i = 0; i < size(); ++i)
            total += neurons[i].loss(other.neurons[i]);
        return total / size();
    }

    void fft(bool inverse = false)
    {
        size_t N = size();
        if (N == 0) return;
        std::vector<std::complex<double>> signal(N);
        for (size_t i = 0; i < N; ++i) signal[i] = neurons[i].z;
        double sign = inverse ? 1.0 : -1.0;
        std::vector<std::complex<double>> result(N, {0.0, 0.0});
        for (size_t k = 0; k < N; ++k)
        {
            std::complex<double> sum(0.0, 0.0);
            for (size_t n = 0; n < N; ++n)
            {
                double angle = sign * 2.0 * M_PI * k * n / N;
                sum += signal[n] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            if (inverse) sum /= static_cast<double>(N);
            result[k] = sum;
        }
        for (size_t i = 0; i < N; ++i) neurons[i].z = result[i];
    }

    Layer sigmoid() const { Layer res; for (auto& n : neurons) res.neurons.push_back(n.sigmoid()); return res; }
    Layer tanh() const    { Layer res; for (auto& n : neurons) res.neurons.push_back(n.tanh());    return res; }
    Layer relu() const    { Layer res; for (auto& n : neurons) res.neurons.push_back(n.relu());    return res; }
    Layer normalize() const { Layer res; for (auto& n : neurons) res.neurons.push_back(n.normalize()); return res; }

    double mean_amplitude() const
    {
        if (neurons.empty()) return 0.0;
        double sum = 0.0;
        for (auto& n : neurons) sum += n.amplitude();
        return sum / neurons.size();
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& l)
    {
        os << "[";
        for (size_t i = 0; i < l.size(); ++i) {
            if (i > 0) os << ", ";
            os << l[i];
        }
        os << "]";
        return os;
    }
};

// ===================== Processor 基类 =====================
template <typename Domain>
class Processor
{
public:
    Processor() = default;
    virtual ~Processor() = default;

    virtual Layer decode(const Domain& input) const = 0;
    virtual Domain encode(const Layer& output) const = 0;
    virtual std::vector<Layer> layers() const = 0;
    virtual void apply(const Domain& input, const Domain& target, double learning_rate) = 0;

    // 联想检索
    std::vector<Layer> remind(const Layer& query, size_t top_k = 1) const
    {
        auto all = layers();
        if (all.empty() || top_k == 0) return {};
        std::vector<std::pair<double, Layer>> scored;
        for (const auto& l : all)
        {
            double sim = l.similarity(query);
            double dist = l.distance(query);
            double score = sim - std::abs(dist);
            scored.emplace_back(score, l);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (top_k > scored.size()) top_k = scored.size();
        std::vector<Layer> result;
        for (size_t i = 0; i < top_k; ++i) result.push_back(scored[i].second);
        return result;
    }

    // 推理函数（非虚）：利用记忆优化输入层
    Layer inference(const Layer& input, size_t top_k = 3) const
    {
        auto similar_layers = remind(input, top_k);
        if (similar_layers.empty()) return input;

        std::vector<double> weights;
        double total_weight = 0.0;
        for (const auto& l : similar_layers)
        {
            double sim = input.similarity(l);
            if (sim < 0.0) sim = 0.0;
            weights.push_back(sim);
            total_weight += sim;
        }
        if (total_weight == 0.0) return input;

        Layer result(input.size(), input[0].omega);
        for (size_t i = 0; i < result.size(); ++i)
        {
            std::complex<double> z_sum(0.0, 0.0);
            double offset_sum = 0.0;
            for (size_t k = 0; k < similar_layers.size(); ++k)
            {
                double w = weights[k] / total_weight;
                z_sum += w * similar_layers[k][i].z;
                offset_sum += w * similar_layers[k][i].offset;
            }
            result[i].z = z_sum;
            result[i].offset = offset_sum;
            result[i].omega = input[i].omega;
        }
        return result;
    }

    // 训练：批量数据，内部根据损失和相似度调整维度
    void train(const std::vector<std::pair<Domain, Domain>>& dataset,
               double learning_rate = 0.01, size_t epochs = 100)
    {
        for (size_t e = 0; e < epochs; ++e)
        {
            for (auto& pair : dataset)
                apply(pair.first, pair.second, learning_rate);

            // 每个 epoch 结束后评估一次
            if (!dataset.empty())
            {
                const auto& first = dataset.front();
                Layer pred = decode(first.first);
                Layer target = decode(first.second);
                double loss_val = pred.loss(target);
                double sim = pred.similarity(target);
                double dist = pred.distance(target);

                if (should_expand(loss_val, sim, dist))
                    expand();
                else if (should_compress(loss_val, sim, dist))
                    compress();
            }
        }
    }

    // 主处理流程：decode -> inference -> predict
    Domain process(const Domain& input)
    {
        Layer current = decode(input);
        Layer refined = inference(current);
        return predict(refined);
    }

    virtual Domain predict(const Layer& output) const
    {
        auto best = remind(output, 1);
        if (best.empty()) return encode(output);
        return encode(best[0]);
    }

    virtual Domain predict(const Layer& output, double creative) const
    {
        auto best = remind(output, 1);
        if (best.empty()) return encode(output);
        Domain result = encode(best[0]);
        double top_sim = output.similarity(best[0]);
        if (top_sim < creative)
        {
            auto top3 = remind(output, 3);
            std::ostringstream oss;
            for (size_t i = 0; i < top3.size(); ++i)
            {
                if (i > 0) oss << "/";
                oss << encode(top3[i]);
            }
            return oss.str() + "?";
        }
        return result;
    }

    Domain predicate(const Layer& output) { return encode(output); }

protected:
    virtual void expand() {}
    virtual void compress() {}
    virtual bool should_expand(double loss, double sim, double dist) const
    {
        return loss > 0.2 && sim < 0.5;
    }
    virtual bool should_compress(double loss, double sim, double dist) const
    {
        return loss < 0.001 && sim > 0.99;
    }
};

// ===================== ConversationProcessor =====================
class ConversationProcessor : public Processor<std::string>
{
private:
    std::unordered_map<std::string, Layer> embeddings;
    Layer dialog_relation;
    size_t embedding_dim;
    double default_omega;

    void expand_all()
    {
        ++embedding_dim;
        for (auto& kv : embeddings)
            kv.second.neurons.emplace_back(Neuron::random(default_omega));
        dialog_relation.neurons.emplace_back(Neuron::random(default_omega));
    }

    void compress_all()
    {
        if (embedding_dim <= 1) return;
        --embedding_dim;
        for (auto& kv : embeddings)
            if (!kv.second.neurons.empty()) kv.second.neurons.pop_back();
        if (!dialog_relation.neurons.empty()) dialog_relation.neurons.pop_back();
    }

    void expand() override { expand_all(); }
    void compress() override { compress_all(); }

public:
    ConversationProcessor(size_t dim = 4, double omega = 1.0)
        : embedding_dim(dim), default_omega(omega)
    {
        dialog_relation = Layer(dim, omega);
        for (size_t i = 0; i < dim; ++i)
            dialog_relation[i] = Neuron::random(omega);
    }

    void set_embedding(const std::string& word, const Layer& layer)
    {
        embeddings[word] = layer;
    }

    // 实现纯虚函数
    Layer decode(const std::string& input) const override
    {
        std::istringstream iss(input);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) words.push_back(word);
        if (words.empty()) return Layer(embedding_dim, default_omega);
        Layer combined = embeddings.at(words[0]);
        for (size_t i = 1; i < words.size(); ++i)
            combined = combined.bind(embeddings.at(words[i]));
        return combined;
    }

    std::string encode(const Layer& output) const override
    {
        if (embeddings.empty()) return "";
        double best_sim = -std::numeric_limits<double>::max();
        std::string best_word;
        for (const auto& kv : embeddings)
        {
            double sim = output.similarity(kv.second);
            if (sim > best_sim) { best_sim = sim; best_word = kv.first; }
        }
        return best_word;
    }

    std::vector<Layer> layers() const override
    {
        std::vector<Layer> res;
        for (const auto& kv : embeddings) res.push_back(kv.second);
        return res;
    }

    void apply(const std::string& input, const std::string& target, double lr) override
    {
        if (embeddings.find(input) == embeddings.end() ||
            embeddings.find(target) == embeddings.end())
            return;

        Layer& Q = embeddings[input];
        Layer& A = embeddings[target];

        Layer pred = Q.bind(dialog_relation);
        for (size_t i = 0; i < pred.size(); ++i)
        {
            std::complex<double> diff = pred[i].z - A[i].z;
            std::complex<double> gradQ = diff * std::conj(dialog_relation[i].z);
            std::complex<double> gradR = diff * std::conj(Q[i].z);
            std::complex<double> gradA = diff;

            Q[i].z -= lr * gradQ;
            dialog_relation[i].z -= lr * gradR;
            A[i].z -= lr * gradA;
        }
    }

    // 对话专用的 process：绑定 dialog_relation 并使用 inference 增强
    std::string process(const std::string& input)
    {
        Layer current = decode(input);
        Layer answer = current.bind(dialog_relation);
        Layer refined = inference(answer);
        return encode(refined);
    }

    // 便捷训练接口
    void train_dialog_set(const std::vector<std::pair<std::string, std::string>>& dialogs,
                          double lr = 0.01, size_t epochs = 200)
    {
        // 转换为基类需要的格式
        std::vector<std::pair<std::string, std::string>> dataset = dialogs;
        train(dataset, lr, epochs);
    }
};

} // namespace neuron