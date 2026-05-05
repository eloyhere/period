#pragma once
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <string>
#include <iostream>
#include <sstream>
#include <optional>
#include <numeric>
#include <functional>
#include <unordered_set>
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
        std::complex<double> value;

        Neuron(): value(std::polar(1.0, 0.0)) {}
        Neuron(double amplitude, double phi ): value(std::polar(amplitude, phi)) {}

        explicit Neuron(const std::complex<double> &value): value(value) {}

        double amplitude() const { return std::abs(this-> value); }
        double phi() const { return std::arg(this->value); }
        double real() const { return this->value.real(); }
        double imaginary() const { return this->value.imag(); }

        void setAmplitude(const double& amplitude) { 
            this->value = std::polar(amplitude, this->phi());
        }
        void setPhi(const double& phi) { 
            this->value = std::polar(this-> amplitude(), phi);
        }

        Neuron operator+(const Neuron& other) const {
            return Neuron(this->value + other.value);
        }
        Neuron operator-(const Neuron& other) const {
            return Neuron(this->value - other.value);
        }
        Neuron operator*(const Neuron& other) const
        {
            return Neuron(this-> value * other.value);
        }
        Neuron operator/(const Neuron& other) const
        {
            if (other.value == std::polar(0.0, 0.0)) {
                throw std::runtime_error("Devision with zero.");
            }
            return Neuron(this->value / other.value);
        }

        Neuron operator*(double k) const { 
            return Neuron(this->value * k);
        }
        Neuron operator/(double k) const { 
            return Neuron(this->value / k);
        }
        friend Neuron operator*(double k, const Neuron& n) { 
            return n * k;
        }

        Neuron operator-() const { return Neuron(-this->value); }
        Neuron operator~() const { return Neuron(std::conj(this->value)); }


        double similarity(const Neuron& other) const
        {
            if (amplitude() == 0.0 || other.amplitude() == 0.0)
                return 0.0;
            return (std::conj(this->value) * other.value).real() / (amplitude() * other.amplitude());
        }

        double distance(const Neuron& other) const
        {
            if (amplitude() == 0.0 || other.amplitude() == 0.0)
                return 0.0;
            return (std::conj(this->value) * other.value).imag() / (amplitude() * other.amplitude());
        }

        double loss(const Neuron& target) const
        {
            std::complex<double> diff = this->value - target.value;
            return fix(std::norm(diff));
        }

        Neuron forward(const std::complex<double>& neighbor_sum, double self_weight = 0.5) const
        {
            return Neuron(self_weight * this->value + (1.0 - self_weight) * neighbor_sum);
        }

        std::pair<std::complex<double>, std::complex<double>> backward(const std::complex<double>& grad_output, const std::complex<double>& neighbor_sum, double self_weight = 0.5) const
        {
            std::complex<double> grad_self = self_weight * grad_output;
            std::complex<double> grad_neighbor = (1.0 - self_weight) * grad_output;
            return { grad_self, grad_neighbor };
        }

        void update(double rate, const std::complex<double>& gradient) { 
            this->value -= rate * gradient;
        }

        Neuron conjugate() const { 
            return Neuron(std::conj(this->value));
        }
        Neuron inverse() const { 
            return Neuron(1.0 / this->value);
        }
        Neuron normalize() const
        {
            double a = amplitude();
            return a == 0.0 ? *this : Neuron(this->value / a);
        }
        double magnitude() const { return amplitude(); }
        double phase() const { return phi(); }

        Neuron sigmoid() const
        {
            double a = amplitude();
            double s = 1.0 / (1.0 + std::exp(-a));
            return Neuron(s, phi());
        }
        Neuron tanh() const
        {
            double a = amplitude();
            double t = std::tanh(a);
            return Neuron(t, phi());
        }
        Neuron relu() const
        {
            double a = amplitude();
            return Neuron(a > 0.0 ? a : 0.0, phi());
        }
        Neuron exp() const { 
            return Neuron(std::exp(this->value));
        }
        Neuron log() const { 
            return Neuron(std::log(this->value));
        }
        Neuron pow(double exponent) const { 
            return Neuron(std::pow(this->value, exponent));
        }

        bool operator==(const Neuron& o) const { 
            return this->value == o.value;
        }
        bool operator!=(const Neuron& o) const { return !(*this == o); }
        bool operator<(const Neuron& o) const { return amplitude() < o.amplitude(); }
        bool operator>(const Neuron& o) const { return amplitude() > o.amplitude(); }
        bool operator<=(const Neuron& o) const { return amplitude() <= o.amplitude(); }
        bool operator>=(const Neuron& o) const { return amplitude() >= o.amplitude(); }

        friend std::ostream& operator<<(std::ostream& os, const Neuron& n)
        {
            os << "[amplitude: " << n.amplitude() << ", phi: " << n.phi() << "] ";
            return os;
        }

        static Neuron random(double omega = 1.0, double offset = 0.0)
        {
            static std::mt19937 rng(std::random_device{}());
            static std::uniform_real_distribution<double> ampDist(0.0, 1.0);
            static std::uniform_real_distribution<double> phiDist(0.0, 2.0 * pi);
            return Neuron(ampDist(rng), phiDist(rng));
        }
    };

    template <typename Domain>
    class Processor {
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

        std::vector<Neuron> disseminate(const std::vector<size_t>& sources, int strength) const {
            const size_t n = network.size();
            std::vector<Neuron> activated(n, Neuron(0.0, 0.0));
            if (strength <= 0 || n == 0) return activated;

            std::vector<bool> visited(n, false);
            std::queue<std::pair<size_t, int>> q;
            for (size_t idx : sources) {
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
                for (const auto& [neighbor, weight] : link[curr]) {
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

        double similar(const Processor<Domain> other) const {
            const size_t n = std::min(network.size(), other.network.size());
            if (n == 0) return 0.0;
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
                sum += network[i].similarity(other.network[i]);
            return sum / n;
        }

        double distance(const Processor<Domain>& other) const {
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
            if (n == 0 || neurons.empty()) return std::vector<Neuron>(n, Neuron(0.0, 0.0));

            std::vector<std::pair<double, size_t>> amp_idx;
            size_t m = std::min(neurons.size(), n);
            for (size_t i = 0; i < m; ++i) {
                amp_idx.emplace_back(neurons[i].amplitude(), i);
            }
            size_t num_sources = std::min(size_t(3), amp_idx.size());
            std::partial_sort(amp_idx.begin(), amp_idx.begin() + num_sources, amp_idx.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            std::vector<size_t> source_indices;
            for (size_t t = 0; t < num_sources; ++t) {
                size_t idx = amp_idx[t].second;
                if (idx < n)
                    source_indices.push_back(idx);
                else if (n > 0 && source_indices.empty())
                    source_indices.push_back(0);
            }
            if (source_indices.empty() && n > 0)
                source_indices.push_back(0);

            auto activated = disseminate(source_indices, 3);

            std::vector<std::pair<double, size_t>> scores;
            for (size_t i = 0; i < n; ++i) {
                if (activated[i].amplitude() == 0.0) continue;
                if (i >= neurons.size()) break;
                double sim = activated[i].similarity(neurons[i]);
                double dist = activated[i].distance(neurons[i]);
                double score = sim - std::abs(dist);
                if (score > 0.0) scores.emplace_back(score, i);
            }

            std::vector<Neuron> result(n, Neuron(0.0, 0.0));
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
            if (neurons.size() == 0) return;
            const double lr = 0.2;
            for (size_t i = 0; i < neurons.size(); ++i) {
                if (i >= network.size()) {
                    this-> network.push_back(neurons[i]);
                }
                else {
                    this->network[i].value = (1.0 - lr) * this->network[i].value + lr * neurons[i].value;
                }
            }
            relink();
            prune();
        }
        std::vector<Neuron> infer(std::vector<Neuron> neurons) const {
            const size_t n = network.size();
            if (n == 0 || neurons.size() == 0) return neurons;

            size_t top_k = std::min(size_t(5), n);
            auto mem = remind(neurons, top_k);
            std::vector<Neuron> result = neurons;
            result.resize(n, Neuron(0.0, 0.0));
            for (size_t i = 0; i < n; ++i) {
                if (i >= neurons.size()) break;
                if (mem[i].amplitude() > 0.0) {
                    double w = mem[i].similarity(neurons[i]);
                    w = std::max(0.0, std::min(1.0, w));
                    result[i].value = (1.0 - w) * neurons[i].value + w * mem[i].value;
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
            final_mem.resize(network.size(), Neuron(0.0, 0.0));
            for (size_t i = 0; i < network.size(); ++i) {
                if (i >= final_mem.size()) break;
                if (top_mem[i].amplitude() > 0.0) {
                    double w = top_mem[i].similarity(decoded[i % decoded.size()]);
                    w = std::max(0.0, std::min(1.0, w));
                    final_mem[i].value = (1.0 - w) * final_mem[i].value + w * top_mem[i].value;
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

        virtual void train(const std::initializer_list<std::string>& data, double threshold, int strength, size_t epoch = 1000) = 0;

        virtual double loss() const = 0;
    };

    class Period : public Processor<std::string> {
    public:
        Period() = default;

       
        void train(const std::initializer_list<std::string>& data,
            double threshold, int strength, size_t epoch = 1000) override {
            concepts.clear();
            name.clear();
            std::vector<std::string> ordered;
            for (const auto& s : data) {
                if (name.find(s) == name.end()) {
                    name[s] = ordered.size();
                    ordered.push_back(s);
                }
            }
            concepts = std::move(ordered);
            const size_t N = concepts.size();
            network.resize(N);
            const double delta = 2.0 * pi / N;
            for (size_t i = 0; i < N; ++i) {
                double amp = std::abs(std::sin(2.0 * pi * static_cast<double>(i) / N));
                if (amp < 1e-6) amp = 1e-6;
                network[i] = Neuron(amp, i * delta);
            }
        }
        std::vector<std::string> predict(std::vector<std::string> domains) override {
            std::vector<std::string> input(domains);
            auto neurons = decode(input);
            if (neurons.empty()) return {};

            Neuron result(0.0, 0.0);
            if (input.size() == 1) {
                result = neurons[0];
            }
            else if (input.size() == 2) {
                double phi = std::fmod(neurons[0].phi() + neurons[1].phi(), 2.0 * pi);
                if (phi < 0) phi += 2.0 * pi;
                result = Neuron(1.0, phi);
            }
            else if (input.size() == 3) {
                double p0 = neurons[0].phi();
                double p1 = neurons[1].phi();
                double p2 = neurons[2].phi();

                std::vector<double> sorted = { p0, p1, p2 };
                std::sort(sorted.begin(), sorted.end());
                double gap1 = sorted[1] - sorted[0];
                double gap2 = sorted[2] - sorted[1];

                const double eps = 0.1; 
                bool isTriple = false;  
                bool isTriangle = false; 
                if (std::abs(gap1 - 2.0 * pi / 12.0) < eps && std::abs(gap2 - 2.0 * pi / 12.0) < eps) {
                    isTriple = true;
                    double midPhi = sorted[1]; 
                    result = Neuron(1.0, midPhi);
                }
                else if (std::abs(gap1 - 2.0 * pi / 3.0) < eps && std::abs(gap2 - 2.0 * pi / 3.0) < eps) {
                    isTriangle = true;
                    double sumPhi = std::fmod(p0 + p1 + p2, 2.0 * pi);
                    if (sumPhi < 0) sumPhi += 2.0 * pi;
                    result = Neuron(1.0, sumPhi);
                }

                if (!isTriple && !isTriangle) {
                    double phi = std::fmod(neurons[0].phi() - neurons[1].phi() + neurons[2].phi(), 2.0 * pi);
                    if (phi < 0) phi += 2.0 * pi;
                    result = Neuron(1.0, phi);
                }
            }
            else {
                result = neurons[0];
            }

            std::string best = find_closest(result);
            return { best };
        }

        std::vector<Neuron> decode(const std::string& d) const override {
            auto it = name.find(d);
            if (it != name.end())
                return { network.at(it->second) };
            throw std::runtime_error("Unknown concept: " + d);
        }

        std::vector<Neuron> decode(std::vector<std::string> ds) const override {
            std::vector<Neuron> res;
            for (const auto& d : ds) {
                auto it = name.find(d);
                if (it != name.end())
                    res.push_back(network.at(it->second));
                else
                    res.push_back(Neuron(0.0, 0.0));
            }
            return res;
        }
        std::vector<std::string> encode(std::vector<Neuron> ns) const override {
            std::vector<std::string> res;
            for (const auto& n : ns) {
                if (n.amplitude() < 1e-6)
                    res.push_back("?");
                else
                    res.push_back(find_closest(n));
            }
            return res;
        }

        double loss() const override { return 0.0; }

        friend std::ostream& operator<<(std::ostream& stream, const Period& processor) {
            stream << "Period with " << processor.concepts.size() << " concepts";
            return stream;
        }

    private:
        std::vector<std::string> concepts;
        std::unordered_map<std::string, size_t> name;
        std::string find_closest(const Neuron& target) const {
            double best = -2.0;
            std::string best_concept;
            for (const auto& c : concepts) {
                double sim = target.similarity(network.at(name.at(c)));
                if (sim > best) {
                    best = sim;
                    best_concept = c;
                }
            }
            return best_concept;
        }
    };

}; // namespace neuron
