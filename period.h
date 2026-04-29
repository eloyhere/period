#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <queue>
#include <unordered_map>
namespace neuron
{
    constexpr double pi = 3.141592653589793238462643383279502884197169;

    inline double fix(double value, double epsilon = 1e-12)
    {
        return (std::abs(value) < epsilon) ? 0.0 : value;
    }

    inline std::complex<double> fix(const std::complex<double>& z, double epsilon = 1e-12)
    {
        return { fix(z.real(), epsilon), fix(z.imag(), epsilon) };
    }

    class Neuron
    {
    public:
        std::complex<double> z;
        double omega;
        double offset;

        Neuron(double amplitude = 1.0, double omega = 1.0, double phi = 0.0, double offset = 0.0)
            : z(std::polar(amplitude, phi)), omega(omega), offset(offset) {
        }

        explicit Neuron(const std::complex<double>& state,
            double omega = 1.0, double offset = 0.0)
            : z(state), omega(omega), offset(offset) {
        }

        double amplitude() const { return std::abs(z); }
        double phi() const { return std::arg(z); }
        double real() const { return z.real(); }
        double imaginary() const { return z.imag(); }

        void setAmplitude(double a) { z = std::polar(a, phi()); }
        void setPhi(double p) { z = std::polar(amplitude(), p); }

        Neuron operator*(const Neuron& o) const
        {
            return Neuron(z * o.z, omega, offset + o.offset);
        }

        Neuron operator/(const Neuron& o) const
        {
            if (o.amplitude() == 0.0)
                throw std::domain_error("Division by zero-amplitude neuron");
            return Neuron(z / o.z, omega, offset - o.offset);
        }

        Neuron operator*(double k) const { return Neuron(z * k, omega, offset * k); }
        Neuron operator/(double k) const { return Neuron(z / k, omega, offset / k); }
        friend Neuron operator*(double k, const Neuron& n) { return n * k; }

        Neuron operator+(const Neuron& o) const { return Neuron(z + o.z, omega, offset + o.offset); }
        Neuron operator-(const Neuron& o) const { return Neuron(z - o.z, omega, offset - o.offset); }

        Neuron operator-() const { return Neuron(-z, omega, -offset); }
        Neuron operator~() const { return Neuron(std::conj(z), omega, offset); }

        double sine(double x) const { return fix(amplitude() * std::sin(omega * x + phi()) + offset); }
        double cosine(double x) const { return fix(amplitude() * std::cos(omega * x + phi()) + offset); }

        std::complex<double> gradient() const { return fix(std::complex<double>(-imaginary(), real())); }

        double similarity(const Neuron& other) const
        {
            if (amplitude() == 0.0 || other.amplitude() == 0.0)
                return 0.0;
            return (std::conj(z) * other.z).real() / (amplitude() * other.amplitude());
        }

        double distance(const Neuron& other) const
        {
            if (amplitude() == 0.0 || other.amplitude() == 0.0)
                return 0.0;
            return (std::conj(z) * other.z).imag() / (amplitude() * other.amplitude());
        }

        double loss(const Neuron& target) const
        {
            std::complex<double> diff = z - target.z;
            return fix(std::norm(diff));
        }

        Neuron forward(const std::complex<double>& neighbor_sum, double self_weight = 0.5) const
        {
            std::complex<double> new_z = self_weight * z + (1.0 - self_weight) * neighbor_sum;
            return Neuron(new_z, omega, offset);
        }

        std::pair<std::complex<double>, std::complex<double>> backward(const std::complex<double>& grad_output, const std::complex<double>& neighbor_sum, double self_weight = 0.5) const
        {
            std::complex<double> grad_self = self_weight * grad_output;
            std::complex<double> grad_neighbor = (1.0 - self_weight) * grad_output;
            return { grad_self, grad_neighbor };
        }

        void update(double lr, const std::complex<double>& grad_z) { z -= lr * grad_z; }

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

        bool operator==(const Neuron& o) const { return z == o.z && omega == o.omega && offset == o.offset; }
        bool operator!=(const Neuron& o) const { return !(*this == o); }
        bool operator<(const Neuron& o) const { return amplitude() < o.amplitude(); }
        bool operator>(const Neuron& o) const { return amplitude() > o.amplitude(); }
        bool operator<=(const Neuron& o) const { return amplitude() <= o.amplitude(); }
        bool operator>=(const Neuron& o) const { return amplitude() >= o.amplitude(); }

        friend std::ostream& operator<<(std::ostream& os, const Neuron& n)
        {
            os << "[amplitude: " << n.amplitude() << ", phi: " << n.phi()
                << ", omega: " << n.omega << ", offset: " << n.offset << "]";
            return os;
        }

        static Neuron random(double omega = 1.0, double offset = 0.0)
        {
            static std::mt19937 rng(std::random_device{}());
            static std::uniform_real_distribution<double> ampDist(0.0, 1.0);
            static std::uniform_real_distribution<double> phiDist(0.0, 2.0 * pi);
            return Neuron(ampDist(rng), omega, phiDist(rng), offset);
        }
    };

    template <typename Domain>
    class Gragh {
    protected:
        std::vector<Neuron> network;
        std::vector<std::unordered_map<size_t, double>> link;

        void relink() {
            const size_t n = network.size();
            link.assign(n, {});
            if (n == 0) return;
            for (size_t i = 0; i < n; ++i) {
                std::vector<std::pair<double, size_t>> scores;
                for (size_t j = 0; j < n; ++j) {
                    if (i == j) continue;
                    double sim = network[i].similarity(network[j]);
                    double dist = network[i].distance(network[j]);
                    double score = sim - std::abs(dist);
                    if (score > 0.0) scores.emplace_back(score, j);
                }
                const size_t k = std::min(scores.size(), size_t(5));
                std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                double sum = 0.0;
                for (size_t t = 0; t < k; ++t) sum += scores[t].first;
                for (size_t t = 0; t < k; ++t) {
                    double w = (sum > 0) ? scores[t].first / sum : 0.0;
                    link[i][scores[t].second] = w;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (auto& [j, w] : link[i]) {
                    if (link[j].find(i) == link[j].end())
                        link[j][i] = w;
                    else
                        link[j][i] = std::max(w, link[j][i]);
                }
            }
        }

    public:
        virtual std::vector<Neuron> decode(const Domain& domain) const = 0;
        virtual std::vector<Neuron> decode(std::vector<Domain> domains) const = 0;

        virtual std::vector<Domain> encode(std::vector<Neuron> neurons) const = 0;

        std::vector<Neuron> disseminate(const std::queue<size_t>& sources, int strength) const {
            const size_t n = network.size();
            std::vector<Neuron> activated(n, Neuron(0.0, 0.0, 0.0, 0.0));
            if (strength <= 0 || n == 0) return activated;

            std::vector<bool> visited(n, false);
            std::queue<std::pair<size_t, int>> q;
            std::queue<size_t> src_copy = sources;
            while (!src_copy.empty()) {
                size_t idx = src_copy.front();
                src_copy.pop();
                if (idx < n && !visited[idx]) {
                    visited[idx] = true;
                    q.emplace(idx, strength);
                }
            }

            while (!q.empty()) {
                auto [curr, cur_strength] = q.front();
                q.pop();
                if (cur_strength <= 0) continue;
                int next_strength = cur_strength - 1;
                auto it = link[curr].begin();
                for (; it != link[curr].end(); ++it) {
                    size_t neighbor = it->first;
                    double weight = it->second;
                    if (weight <= 0.0) continue;
                    if (neighbor < n && !visited[neighbor]) {
                        visited[neighbor] = true;
                        q.emplace(neighbor, next_strength);
                    }
                }
            }

            for (size_t i = 0; i < n; ++i) {
                if (visited[i]) activated[i] = network[i];
            }
            return activated;
        }

        double similar(const Gragh& other) const {
            const size_t n = std::min(network.size(), other.network.size());
            if (n == 0) return 0.0;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
                sum += network[i].similarity(other.network[i]);
            return sum / n;
        }

        double distance(const Gragh& other) const {
            const size_t n = std::min(network.size(), other.network.size());
            if (n == 0) return 0.0;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
                sum += network[i].distance(other.network[i]);
            return sum / n;
        }

        std::vector<Neuron> remind(std::vector<Neuron> neurons) const {
            return remind(neurons, std::max(size_t(1), neurons.size()));
        }
        std::vector<Neuron> remind(std::vector<Neuron> neurons, const size_t& top) const {
            const size_t n = network.size();
            if (n == 0 || neurons.empty()) return std::vector<Neuron>(n, Neuron(0.0, 0.0, 0.0, 0.0));
            std::queue<size_t> sources;
            std::vector<std::pair<double, size_t>> amp_idx;
            size_t m = std::min(neurons.size(), n);
            for (size_t i = 0; i < m; ++i) {
                amp_idx.emplace_back(neurons[i].amplitude(), i);
            }
            size_t num_sources = std::min(size_t(3), amp_idx.size());
            std::partial_sort(amp_idx.begin(), amp_idx.begin() + num_sources, amp_idx.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t t = 0; t < num_sources; ++t) {
                sources.push(amp_idx[t].second);
            }

            auto activated = disseminate(sources, 3);
            std::vector<std::pair<double, size_t>> scores;
            for (size_t i = 0; i < n; ++i) {
                if (activated[i].amplitude() == 0.0) continue;
                if (i >= neurons.size()) break;
                double sim = activated[i].similarity(neurons[i]);
                double dist = activated[i].distance(neurons[i]);
                double score = sim - std::abs(dist);
                if (score > 0.0) scores.emplace_back(score, i);
            }

            std::vector<Neuron> result(n, Neuron(0.0, 0.0, 0.0, 0.0));
            if (scores.empty()) return result;
            size_t k = std::min(top, scores.size());
            std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t t = 0; t < k; ++t) {
                size_t idx = scores[t].second;
                result[idx] = network[idx];
            }
            return result;
        }
        void remember(std::vector<Neuron> neurons) {
            if (neurons.empty()) return;
            const double lr = 0.2;
            for (size_t i = 0; i < neurons.size(); ++i) {
                if (i >= network.size()) {
                    network.push_back(neurons[i]);
                }
                else {
                    network[i].z = (1.0 - lr) * network[i].z + lr * neurons[i].z;
                }
            }
            relink();
            prune();
        }
        std::vector<Neuron> infer(std::vector<Neuron> neurons) const {
            const size_t n = network.size();
            if (n == 0 || neurons.empty()) return neurons;

            size_t top_k = std::min(size_t(5), n);
            auto mem = remind(neurons, top_k); 
            std::vector<Neuron> result = neurons;
            result.resize(n, Neuron(0.0, 0.0, 0.0, 0.0));
            for (size_t i = 0; i < n; ++i) {
                if (i >= neurons.size()) break;
                if (mem[i].amplitude() > 0.0) {
                    double w = mem[i].similarity(neurons[i]); 
                    w = std::max(0.0, std::min(1.0, w));
                    result[i].z = (1.0 - w) * neurons[i].z + w * mem[i].z;
                }
                else {
                    result[i] = neurons[i];
                }
            }
            return result;
        }

        virtual std::vector<Domain> predict(std::vector<Domain> domains) {
            auto decoded = decode(domains);
            auto inferred = infer(decoded);
            size_t top_k = std::min(size_t(5), network.size());
            auto top_mem = remind(decoded, top_k);
            std::vector<Neuron> final_mem = inferred;
            final_mem.resize(network.size(), Neuron(0.0, 0.0, 0.0, 0.0));
            for (size_t i = 0; i < network.size(); ++i) {
                if (i >= final_mem.size()) break;
                if (top_mem[i].amplitude() > 0.0) {
                    double w = top_mem[i].similarity(decoded[i % decoded.size()]);
                    w = std::max(0.0, std::min(1.0, w));
                    final_mem[i].z = (1.0 - w) * final_mem[i].z + w * top_mem[i].z;
                }
            }
            remember(final_mem);
            return encode(final_mem);
        }

        void prune(double threshold = 0.01) {
            std::vector<size_t> to_remove;
            for (size_t i = 0; i < network.size(); ++i) {
                if (network[i].amplitude() < threshold)
                    to_remove.push_back(i);
            }
            if (to_remove.empty()) return;
            std::sort(to_remove.rbegin(), to_remove.rend());
            for (auto idx : to_remove) {
                network.erase(network.begin() + idx);
                link.erase(link.begin() + idx);
            }
            relink();
        }

        std::vector<std::vector<size_t>> group(double threshold = 0.3) const {
            const size_t n = network.size();
            std::vector<bool> visited(n, false);
            std::vector<std::vector<size_t>> groups;

            for (size_t i = 0; i < n; ++i) {
                if (visited[i]) continue;
                std::vector<size_t> group;
                std::queue<size_t> q;
                q.push(i);
                visited[i] = true;

                while (!q.empty()) {
                    size_t curr = q.front(); q.pop();
                    group.push_back(curr);
                    for (const auto& [neighbor, w] : link[curr]) {
                        if (w > threshold && !visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
                groups.push_back(group);
            }
            return groups;
        }

        virtual void train(const std::vector<std::pair<std::vector<Domain>, std::vector<Domain>>>& data, double threshold, int strength, size_t epoch = 1000) = 0;

        virtual double loss() const = 0;
    };
}; // namespace neuron

