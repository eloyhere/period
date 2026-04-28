#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>
namespace period
{
inline double fix(double value, double epsilon = 1e-12)
{
    return (std::abs(value) < epsilon) ? 0.0 : value;
}

inline std::complex<double> fix(const std::complex<double> &z, double epsilon = 1e-12)
{
    return {fix(z.real(), epsilon), fix(z.imag(), epsilon)};
}

class Neuron
{
  public:
    std::complex<double> z;
    double omega;
    double offset;

    Neuron(double amplitude = 1.0, double omega = 1.0, double phi = 0.0, double offset = 0.0)
        : z(std::polar(amplitude, phi)), omega(omega), offset(offset) {}

    explicit Neuron(const std::complex<double> &state,
                    double omega = 1.0, double offset = 0.0)
        : z(state), omega(omega), offset(offset) {}

    double amplitude() const { return std::abs(z); }
    double phi() const { return std::arg(z); }
    double real() const { return z.real(); }
    double imag() const { return z.imag(); }

    void setAmplitude(double a) { z = std::polar(a, phi()); }
    void setPhi(double p) { z = std::polar(amplitude(), p); }

    Neuron operator*(const Neuron &o) const
    {
        return Neuron(z * o.z, omega, offset + o.offset);
    }

    Neuron operator/(const Neuron &o) const
    {
        if (o.amplitude() == 0.0)
            throw std::domain_error("Division by zero-amplitude neuron");
        return Neuron(z / o.z, omega, offset - o.offset);
    }

    Neuron operator*(double k) const { return Neuron(z * k, omega, offset * k); }
    Neuron operator/(double k) const { return Neuron(z / k, omega, offset / k); }
    friend Neuron operator*(double k, const Neuron &n) { return n * k; }

    Neuron operator+(const Neuron &o) const { return Neuron(z + o.z, omega, offset + o.offset); }
    Neuron operator-(const Neuron &o) const { return Neuron(z - o.z, omega, offset - o.offset); }

    Neuron operator-() const { return Neuron(-z, omega, -offset); }
    Neuron operator~() const { return Neuron(std::conj(z), omega, offset); }

    double sine(double x) const { return fix(amplitude() * std::sin(omega * x + phi()) + offset); }
    double cosine(double x) const { return fix(amplitude() * std::cos(omega * x + phi()) + offset); }

    std::complex<double> gradient() const { return fix(std::complex<double>(-imag(), real())); }

    double similarity(const Neuron &other) const
    {
        if (amplitude() == 0.0 || other.amplitude() == 0.0)
            return 0.0;
        return (std::conj(z) * other.z).real() / (amplitude() * other.amplitude());
    }

    double distance(const Neuron &other) const
    {
        if (amplitude() == 0.0 || other.amplitude() == 0.0)
            return 0.0;
        return (std::conj(z) * other.z).imag() / (amplitude() * other.amplitude());
    }

    double loss(const Neuron &target) const
    {
        std::complex<double> diff = z - target.z;
        return fix(std::norm(diff));
    }

    void update(double lr, const std::complex<double> &grad_z) { z -= lr * grad_z; }

    Neuron conjugate() const { return Neuron(std::conj(z), omega, offset); }
    Neuron inverse() const { return Neuron(1.0 / z, omega, offset); }
    Neuron normalize() const
    {
        double a = amplitude();
        return a == 0.0 ? *this : Neuron(z / a, omega, offset);
    }
    double magnitude() const { return amplitude(); }
    double phase() const { return phi(); }

    Neuron sigmoid() const
    {
        double a = amplitude();
        double s = 1.0 / (1.0 + std::exp(-a));
        return Neuron(s, omega, phi(), offset);
    }
    Neuron tanh() const
    {
        double a = amplitude();
        double t = std::tanh(a);
        return Neuron(t, omega, phi(), offset);
    }
    Neuron relu() const
    {
        double a = amplitude();
        return Neuron(a > 0.0 ? a : 0.0, omega, phi(), offset);
    }
    Neuron exp() const { return Neuron(std::exp(z), omega, offset); }
    Neuron log() const { return Neuron(std::log(z), omega, offset); }
    Neuron pow(double exponent) const { return Neuron(std::pow(z, exponent), omega, offset); }

    bool operator==(const Neuron &o) const { return z == o.z && omega == o.omega && offset == o.offset; }
    bool operator!=(const Neuron &o) const { return !(*this == o); }
    bool operator<(const Neuron &o) const { return amplitude() < o.amplitude(); }
    bool operator>(const Neuron &o) const { return amplitude() > o.amplitude(); }
    bool operator<=(const Neuron &o) const { return amplitude() <= o.amplitude(); }
    bool operator>=(const Neuron &o) const { return amplitude() >= o.amplitude(); }

    friend std::ostream &operator<<(std::ostream &os, const Neuron &n)
    {
        os << "[amplitude: " << n.amplitude() << ", phi: " << n.phi()
           << ", omega: " << n.omega << ", offset: " << n.offset << "]";
        return os;
    }

    static Neuron random(double omega = 1.0, double offset = 0.0)
    {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> ampDist(0.0, 1.0);
        static std::uniform_real_distribution<double> phiDist(0.0, 2.0 * M_PI);
        return Neuron(ampDist(rng), omega, phiDist(rng), offset);
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;
    std::vector<Neuron> weights;

    Layer() = default;

    Layer(size_t size, double default_omega = 1.0, bool init_weights = true) {
        neurons.reserve(size);
        for (size_t i = 0; i < size; ++i)
            neurons.emplace_back(1.0, default_omega, 0.0, 0.0);
        if (init_weights) {
            weights.reserve(size);
            for (size_t i = 0; i < size; ++i)
                weights.emplace_back(Neuron::random(default_omega, 0.0));
        }
    }

    explicit Layer(const std::vector<Neuron>& n) : neurons(n) {}

    Layer(const std::vector<Neuron>& n, const std::vector<Neuron>& w)
        : neurons(n), weights(w) {}

    size_t size() const { return neurons.size(); }
    Neuron& operator[](size_t i) { return neurons[i]; }
    const Neuron& operator[](size_t i) const { return neurons[i]; }

    Layer dft() const {
        size_t N = size();
        Layer freqLayer;
        freqLayer.neurons.reserve(N);
        const double twoPi = 2.0 * M_PI;
        for (size_t k = 0; k < N; ++k) {
            std::complex<double> sum(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                double angle = -twoPi * k * n / N;
                sum += neurons[n].z * std::polar(1.0, angle);
            }
            double freqOmega = (k <= N/2) ? (twoPi * k / N) : (twoPi * (k - N) / N);
            freqLayer.neurons.emplace_back(sum, freqOmega, 0.0);
        }
        return freqLayer;
    }

    Layer idft() const {
        size_t N = size();
        Layer timeLayer;
        timeLayer.neurons.reserve(N);
        const double twoPi = 2.0 * M_PI;
        for (size_t n = 0; n < N; ++n) {
            std::complex<double> sum(0.0, 0.0);
            for (size_t k = 0; k < N; ++k) {
                double angle = twoPi * k * n / N;
                sum += neurons[k].z * std::polar(1.0, angle);
            }
            sum /= static_cast<double>(N);
            timeLayer.neurons.emplace_back(sum, 1.0, 0.0);
        }
        return timeLayer;
    }

    Layer fft() const {
        size_t N = size();
        if (N == 0 || (N & (N - 1)) != 0)
            throw std::invalid_argument("FFT requires size to be a power of two.");
        if (N == 1) {
            Layer freq;
            freq.neurons.push_back(neurons[0]);
            return freq;
        }
        Layer even, odd;
        for (size_t i = 0; i < N; i += 2) {
            even.neurons.push_back(neurons[i]);
            odd.neurons.push_back(neurons[i + 1]);
        }
        Layer evenF = even.fft();
        Layer oddF = odd.fft();
        Layer result(N, 1.0, false);
        for (size_t k = 0; k < N / 2; ++k) {
            std::complex<double> twiddle = std::polar(1.0, -2.0 * M_PI * k / N);
            std::complex<double> t = twiddle * oddF[k].z;
            result.neurons[k] = Neuron(evenF[k].z + t, 1.0, 0.0);
            result.neurons[k + N / 2] = Neuron(evenF[k].z - t, 1.0, 0.0);
        }
        return result;
    }

    Layer ifft() const {
        size_t N = size();
        if (N == 0 || (N & (N - 1)) != 0)
            throw std::invalid_argument("IFFT requires size to be a power of two.");
        Layer conjFreq;
        for (size_t k = 0; k < N; ++k)
            conjFreq.neurons.push_back(~neurons[k]);
        Layer time = conjFreq.fft();
        for (size_t n = 0; n < N; ++n) {
            time.neurons[n].z = std::conj(time.neurons[n].z) / static_cast<double>(N);
        }
        return time;
    }

    Layer forward(const Layer& input) const {
        if (input.size() != weights.size())
            throw std::domain_error("Input size must match weights size.");
        Layer freqInput = input.dft();
        Layer freqOutput(freqInput.size(), 1.0, false);
        for (size_t i = 0; i < freqInput.size(); ++i) {
            freqOutput[i] = freqInput[i] * weights[i];
        }
        return freqOutput.idft();
    }

    std::pair<Layer, Layer> backward(const Layer& grad_output, const Layer& input) const {
        size_t N = weights.size();
        if (input.size() != N || grad_output.size() != N)
            throw std::domain_error("Size mismatch in backward.");

        Layer dZ = grad_output.dft();
        Layer X = input.dft();

        Layer grad_W(N, 1.0, false);
        for (size_t k = 0; k < N; ++k) {
            grad_W[k] = Neuron(dZ[k].z * std::conj(X[k].z), 1.0, 0.0);
        }

        Layer grad_X_freq(N, 1.0, false);
        for (size_t k = 0; k < N; ++k) {
            grad_X_freq[k] = Neuron(std::conj(weights[k].z) * dZ[k].z, 1.0, 0.0);
        }
        Layer grad_input = grad_X_freq.idft();

        return {grad_input, grad_W};
    }

    Layer gradient() const {
        Layer grad(neurons.size(), 1.0, false);
        for (size_t i = 0; i < neurons.size(); ++i) {
            grad.neurons[i] = Neuron(neurons[i].gradient(), neurons[i].omega, 0.0);
        }
        return grad;
    }

    double similarity(const Layer& other) const {
        if (size() != other.size() || size() == 0) return 0.0;
        double sum = 0.0, wsum = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            double ampA = neurons[i].amplitude(), ampB = other[i].amplitude();
            if (ampA == 0.0 || ampB == 0.0) continue;
            double sim = neurons[i].similarity(other[i]);
            double w = ampA * ampB;
            sum += sim * w;
            wsum += w;
        }
        return (wsum == 0.0) ? 0.0 : (sum / wsum);
    }

    double distance(const Layer& other) const {
        if (size() != other.size() || size() == 0) return 0.0;
        double sum = 0.0, wsum = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            double ampA = neurons[i].amplitude(), ampB = other[i].amplitude();
            if (ampA == 0.0 || ampB == 0.0) continue;
            double dist = neurons[i].distance(other[i]);
            double w = ampA * ampB;
            sum += dist * w;
            wsum += w;
        }
        return (wsum == 0.0) ? 0.0 : (sum / wsum);
    }

    double loss(const Layer& target) const {
        if (size() != target.size()) return 1e9;
        double total = 0.0;
        for (size_t i = 0; i < size(); ++i)
            total += neurons[i].loss(target[i]);
        return total / size();
    }

    Layer operator+(const Layer& o) const {
        checkSize(o);
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] + o[i]);
        return res;
    }
    Layer operator-(const Layer& o) const {
        checkSize(o);
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] - o[i]);
        return res;
    }
    Layer operator*(const Layer& o) const {
        checkSize(o);
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] * o[i]);
        return res;
    }
    Layer operator/(const Layer& o) const {
        checkSize(o);
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] / o[i]);
        return res;
    }
    Layer operator*(double k) const {
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] * k);
        return res;
    }
    Layer operator/(double k) const {
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(neurons[i] / k);
        return res;
    }
    friend Layer operator*(double k, const Layer& l) { return l * k; }

    Layer operator-() const {
        Layer res;
        for (size_t i = 0; i < size(); ++i)
            res.neurons.push_back(-neurons[i]);
        return res;
    }

    bool operator==(const Layer& o) const {
        return size() == o.size() && neurons == o.neurons;
    }
    bool operator!=(const Layer& o) const { return !(*this == o); }

    bool operator<(const Layer& o) const {
        return totalAmplitude() < o.totalAmplitude();
    }
    bool operator>(const Layer& o) const {
        return totalAmplitude() > o.totalAmplitude();
    }
    bool operator<=(const Layer& o) const {
        return totalAmplitude() <= o.totalAmplitude();
    }
    bool operator>=(const Layer& o) const {
        return totalAmplitude() >= o.totalAmplitude();
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& l) {
        os << "Layer[" << l.size() << " neurons]";
        return os;
    }

    double totalAmplitude() const {
        double sum = 0.0;
        for (auto& n : neurons) sum += n.amplitude();
        return sum;
    }

    static Layer random(size_t n, double omega = 1.0, double offset = 0.0) {
        Layer l;
        for (size_t i = 0; i < n; ++i)
            l.neurons.push_back(Neuron::random(omega, offset));
        return l;
    }

    void updateWeights(double lr, const Layer& grad_W) {
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i].update(lr, grad_W[i].z);
    }

    void updateNeurons(double lr, const Layer& grad) {
        for (size_t i = 0; i < neurons.size(); ++i)
            neurons[i].update(lr, grad[i].z);
    }

private:
    void checkSize(const Layer& o) const {
        if (size() != o.size())
            throw std::domain_error("Layer size mismatch");
    }
};

template <typename Domain>
class Processor {
protected:
    std::vector<Layer> memory;
    size_t max_memory;
    size_t trim_amount;
    std::vector<Domain> embed;

public:
    Processor(size_t max_memory = 100, size_t trim_amount = 5)
        : max_memory(max_memory), trim_amount(trim_amount) {}

    virtual ~Processor() = default;

    virtual Layer decode(const Domain& input) const = 0;
    virtual Domain encode(const Layer& output) const = 0;

    std::vector<Layer> remind(const Layer& query, size_t n) const {
        if (memory.empty() || n == 0) return {};
        // 将查询变换到频域，以便在频域比较
        Layer queryFreq = query.dft();
        std::vector<std::pair<double, size_t>> scores;
        scores.reserve(memory.size());
        for (size_t i = 0; i < memory.size(); ++i) {
            if (query.size() != memory[i].size()) {
                scores.emplace_back(-1e9, i);
                continue;
            }
            Layer memFreq = memory[i].dft();          // 记忆项也变换到频域
            double sim = queryFreq.similarity(memFreq);
            double dist = queryFreq.distance(memFreq);
            double score = sim - dist;       // 频域综合得分
            scores.emplace_back(score, i);
        }
        if (n >= memory.size()) {
            return memory;
        }
        std::partial_sort(scores.begin(), scores.begin() + n, scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<Layer> result;
        result.reserve(n);
        for (size_t i = 0; i < n; ++i)
            result.push_back(memory[scores[i].second]);
        return result;
    }

    void remember(const Layer& layer, size_t n = 1) {
        memory.push_back(layer);
        if (memory.size() > max_memory) {
            size_t trim = std::min(n, memory.size() > 0 ? memory.size() - 1 : 0);
            if (trim == 0) return;
            std::vector<std::pair<double, size_t>> scores;
            scores.reserve(memory.size());
            for (size_t i = 0; i < memory.size(); ++i) {
                double sim = layer.similarity(memory[i]);
                double dist = layer.distance(memory[i]);
                double score = sim - dist;
                scores.emplace_back(score, i);
            }
            std::partial_sort(scores.begin(), scores.begin() + trim, scores.end());
            std::vector<size_t> to_remove;
            for (size_t i = 0; i < trim; ++i)
                to_remove.push_back(scores[i].second);
            std::sort(to_remove.rbegin(), to_remove.rend());
            for (auto idx : to_remove)
                memory.erase(memory.begin() + idx);
        }
    }

    Layer inference(const Layer& query, size_t top_k = 5, size_t trim_k = 1) {
        std::vector<Layer> top = remind(query, top_k);
        Layer result;
        if (top.empty()) {
            result = query;
        } else {
            // 将所有候选层变换到频域
            std::vector<Layer> freqTops;
            freqTops.reserve(top.size());
            for (const auto& t : top) {
                freqTops.push_back(t.dft());
            }
            // 在频域做平均（保持信号结构）
            Layer freqAvg = freqTops[0];
            for (size_t i = 1; i < freqTops.size(); ++i) {
                freqAvg = freqAvg + freqTops[i];
            }
            freqAvg = freqAvg / static_cast<double>(freqTops.size());
            // 逆变换回时域作为融合结果
            result = freqAvg.idft();
        }
        // 修剪记忆时同样采用频域评分
        if (trim_k > 0 && !memory.empty()) {
            std::vector<std::pair<double, size_t>> scores;
            scores.reserve(memory.size());
            for (size_t i = 0; i < memory.size(); ++i) {
                if (query.size() != memory[i].size()) {
                    scores.emplace_back(-1e9, i);
                    continue;
                }
                Layer memFreq = memory[i].dft();
                Layer qFreq = query.dft();
                double sim = qFreq.similarity(memFreq);
                double dist = qFreq.distance(memFreq);
                double score = sim - dist;
                scores.emplace_back(score, i);
            }
            size_t remove_cnt = std::min(trim_k, memory.size() > 0 ? memory.size() - 1 : 0);
            if (remove_cnt > 0) {
                std::partial_sort(scores.begin(), scores.begin() + remove_cnt, scores.end());
                std::vector<size_t> to_remove;
                for (size_t i = 0; i < remove_cnt; ++i)
                    to_remove.push_back(scores[i].second);
                std::sort(to_remove.rbegin(), to_remove.rend());
                for (auto idx : to_remove)
                    memory.erase(memory.begin() + idx);
            }
        }
        return result;
    }

    Domain predict(const Layer& layer) {
        size_t top_k = std::min(size_t(5), memory.size());
        auto tops = remind(layer, top_k);
        Layer combined = layer;
        if (!tops.empty()) {
            for (auto& t : tops) combined = combined + t;
            combined = combined / static_cast<double>(1 + tops.size());
        }
        Domain best_domain;
        double best_score = -1e9;
        for (const auto& d : embed) {
            Layer emb_layer = decode(d);
            double sim = combined.similarity(emb_layer);
            double dist = combined.distance(emb_layer);
            double score = sim - dist;
            if (score > best_score) {
                best_score = score;
                best_domain = d;
            }
        }
        Layer query_for_inference = decode(best_domain);
        Layer final_layer = inference(query_for_inference);
        return encode(final_layer);
    }

    Domain process(const Domain& input) {
        Layer current = decode(input);
        Layer result = inference(current);
        remember(result, trim_amount);
        return predict(result);
    }
};

class TrunkProcessor : public Processor<std::string> {
public:
    std::vector<std::string> trunks;

    TrunkProcessor(size_t max_memory = 100, size_t trim_amount = 5,
                   size_t embed_dim = 3, double lr = 0.01, size_t epochs = 500)
        : Processor<std::string>(max_memory, trim_amount)
    {
        trunks = {"甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"};
        // 初始化基类的 embed 列表
        embed = trunks;

        // 为每个天干创建随机嵌入
        for (const auto& t : trunks) {
            Layer layer;
            layer.neurons.reserve(embed_dim);
            for (size_t d = 0; d < embed_dim; ++d)
                layer.neurons.push_back(Neuron::random(1.0, 0.0));
            trunkLayers[t] = layer;
        }

        train(lr, epochs);
    }

    Layer decode(const std::string& input) const override {
        auto it = trunkLayers.find(input);
        if (it != trunkLayers.end())
            return it->second;
        return Layer();
    }

    std::string encode(const Layer& output) const override {
        if (output.size() == 0) return "?";
        double bestScore = -2.0;
        std::string bestStr = "?";
        for (const auto& kv : trunkLayers) {
            double sim = output.similarity(kv.second);
            double dist = output.distance(kv.second);
            double score = sim - std::abs(dist);   // 同时考虑相似度和距离
            if (score > bestScore) {
                bestScore = score;
                bestStr = kv.first;
            }
        }
        return bestStr;
    }

private:
    std::unordered_map<std::string, Layer> trunkLayers;

    void train(double lr, size_t epochs) {
        std::vector<std::tuple<std::string, std::string, std::string>> pairs = {
            {"甲", "己", "己"}, {"乙", "庚", "辛"}, {"丙", "辛", "癸"},
            {"丁", "壬", "乙"}, {"戊", "癸", "丁"}
        };

        for (size_t e = 0; e < epochs; ++e) {
            for (const auto& [a, b, c] : pairs) {
                Layer& A = trunkLayers[a];
                Layer& B = trunkLayers[b];
                Layer& C = trunkLayers[c];

                Layer bind_ab = A * B;

                for (size_t i = 0; i < bind_ab.size(); ++i) {
                    std::complex<double> diff = bind_ab[i].z - C[i].z;
                    std::complex<double> grad_A = diff * std::conj(B[i].z);
                    std::complex<double> grad_B = diff * std::conj(A[i].z);

                    A[i].update(lr, grad_A);
                    B[i].update(lr, grad_B);
                }
            }
        }
    }
};

class InferenceProcessor : public Processor<std::string> {
public:
    InferenceProcessor(
        const std::vector<std::string>& entities,
        const std::vector<std::string>& relations,
        const std::vector<std::tuple<std::string, std::string, std::string>>& triples,
        size_t embed_dim = 10,
        double lr = 0.01,
        size_t epochs = 500,
        size_t max_memory = 100,
        size_t trim_amount = 5
    ) : Processor<std::string>(max_memory, trim_amount) {
        // 初始化基类的 embed 列表
        embed = entities;

        // 初始化实体嵌入
        for (const auto& e : entities)
            entityLayers[e] = Layer::random(embed_dim);

        // 初始化关系嵌入
        for (const auto& r : relations)
            relationLayers[r] = Layer::random(embed_dim);

        // 训练：学习 S * R = O
        for (size_t e = 0; e < epochs; ++e) {
            for (const auto& [s, r, o] : triples) {
                Layer& S = entityLayers[s];
                Layer& R = relationLayers[r];
                Layer& O = entityLayers[o];

                Layer SR = S * R;
                for (size_t i = 0; i < embed_dim; ++i) {
                    std::complex<double> diff = SR[i].z - O[i].z;
                    std::complex<double> grad_S = diff * std::conj(R[i].z);
                    std::complex<double> grad_R = diff * std::conj(S[i].z);
                    S[i].update(lr, grad_S);
                    R[i].update(lr, grad_R);
                }
            }
        }
    }

    // 解码：返回某个实体的嵌入
    Layer decode(const std::string& input) const override {
        auto it = entityLayers.find(input);
        if (it != entityLayers.end()) return it->second;
        return Layer();
    }

    // 编码：找出与输出层最匹配的实体（综合相似度与距离）
    std::string encode(const Layer& output) const override {
        if (output.size() == 0 || embed.empty()) return "?";
        double best = -2.0;
        std::string bestStr = "?";
        for (const auto& name : embed) {
            double sim = output.similarity(entityLayers.at(name));
            double dist = output.distance(entityLayers.at(name));
            double score = sim - std::abs(dist);
            if (score > best) {
                best = score;
                bestStr = name;
            }
        }
        return bestStr;
    }

    // 类比推理：计算 a - b + c （即 a / b * c）
    std::string analogy(const std::string& a,
                        const std::string& b,
                        const std::string& c) {
        Layer la = entityLayers[a];
        Layer lb = entityLayers[b];
        Layer lc = entityLayers[c];
        Layer result = la / lb * lc;   // 复数乘除实现加减
        return encode(result);
    }

private:
    std::unordered_map<std::string, Layer> entityLayers;
    std::unordered_map<std::string, Layer> relationLayers;
};
} // namespace period

