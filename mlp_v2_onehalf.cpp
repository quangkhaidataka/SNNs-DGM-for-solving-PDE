#define N_MAX 7
#define SINE_GORDON2 // Default PDE selection
#define SINE_GORDON1
// #define SINE_GORDON3




#ifdef ALLEN_CAHN
#define eq_name "Allen-Cahn_equation"
#define rdim 1
#define TIME 1.
#define initial_value ArrayXd::Zero(d[j], 1)
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = 1. / (2. + 2. / 5. * x.square().sum())
#define X_sde(s, t, x, w) x + sqrt(2. * (t - s)) * w
#define fn(y) ArrayXd ret = ArrayXd::Zero(1, 1); double phi_r = std::min(4., std::max(-4., y(0))); ret(0) = phi_r - phi_r * phi_r * phi_r
#endif


#ifdef SINE_GORDON1
#define eq_name "sinegordon1"
#define rdim 1
#define TIME 1.
#define initial_value ArrayXd::Zero(d[j], 1) // Initial x = (0, 0, ..., 0)
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = sqrt(1. + x.square().sum()) // New g(x) = sqrt(1 + ||x||^2)
#define X_sde(s, t, x, w) x + sqrt(2. * (t - s)) * w
#define fn(y) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = sin(y(0))
#endif



#ifdef SINE_GORDON2
#define eq_name "sinegordon2"
#define rdim 1
#define TIME 1.
#define initial_value ArrayXd::Zero(d[j], 1) // Initial x = (0, 0, ..., 0)
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = 2/(4+x.square().sum()) // New g(x) = 2/(4+||x||^2)
#define X_sde(s, t, x, w) x + sqrt(2. * (t - s)) * w
#define fn(y) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = sin(y(0))
#endif


#ifdef SINE_GORDON3
#define eq_name "sinegordon3"
#define rdim 1
#define TIME 1.
#define initial_value ArrayXd::Zero(d[j], 1) // Initial x = (0, 0, ..., 0)
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = atan(sqrt(x.square().sum())/2) // New g(x) = arctan(||x||/2)
#define X_sde(s, t, x, w) x + sqrt(2. * (t - s)) * w
#define fn(y) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = sin(y(0))
#endif

#ifdef SEMILINEAR_BS
#define eq_name "Semilinear_Black-Scholes_equation"
#define rdim 1
#define TIME 1.
#define initial_value ArrayXd::Constant(d[j], 1, 50.);
#define g(x) ArrayXd tmp = ArrayXd::Zero(1, 1); tmp(0) = log(0.5 * (1. + x.square().sum()))
#define X_sde(s, t, x, w) x * ((t - s) / 2. + sqrt(t - s) * w).exp()
#define fn(y) ArrayXd ret = ArrayXd::Zero(1, 1); ret(0) = y(0) / (1. + y(0) * y(0))
#endif
#ifdef BS_SYSTEM
#define eq_name "Semilinear_Black-Scholes_system"
#define rdim 2
#define TIME .5
#define initial_value ArrayXd::Constant(d[j], 1, 5.);
#define g(x) ArrayXd tmp = ArrayXd::Zero(2, 1); tmp(0) = 1. / (2. + 2. / 5. * x.square().sum()); tmp(1) = log(0.5 * (1. + x.square().sum()))
#define X_sde(s, t, x, w) x * ((t - s) / 2. + sqrt(t - s) * w).exp()
#define fn(y) ArrayXd ret = ArrayXd::Zero(2, 1); ret(0) = sin(y(1) / 2. / M_PI); ret(1) = (y(1) - y(0)) / (1. + y(0) * y(0) + y(1) * y(1))
#endif
#ifdef PDE_SYSTEM
#define eq_name "Semilinear_PDE_system"
#define rdim 2
#define TIME 1.
#define initial_value ArrayXd::Zero(d[j], 1)
#define g(x) ArrayXd tmp = ArrayXd::Zero(2, 1); tmp(0) = 1. / (2. + 2. / 5. * x.square().sum()); tmp(1) = log(0.5 * (1. + x.square().sum()))
#define X_sde(s, t, x, w) x + sqrt(2. * (t - s)) * w
#define fn(y) ArrayXd ret = ArrayXd::Zero(2, 1); ret(0) = y(1) / (1. + y(1) * y(1)); ret(1) = 2. * y(0) / 3.
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <random>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>
#include </Users/user/eigen-3.3.7/Eigen/Dense>

using Eigen::ArrayXd;

struct mlp_t {
    uint8_t m;
    uint8_t n;
    uint8_t l;
    uint16_t d;
    ArrayXd x;
    double s;
    double t;
    ArrayXd res;
};

ArrayXd f(const ArrayXd &v);
ArrayXd mlp_call(uint8_t m, uint8_t n, uint8_t l, uint16_t d, ArrayXd &x, double s, double t);
ArrayXd ml_picard(uint8_t m, uint8_t n, uint16_t d, ArrayXd &x, double s, double t, bool start_threads);
void mlp_thread(mlp_t &mlp_args);

int main(int argc, char** argv) {
    std::string s = eq_name;
    std::cout << s << std::endl << std::endl << std::setprecision(8);

    std::ofstream out_file;
    out_file.open(s + "_mlp.csv");
    out_file << "d, t, n, ";
    for (uint8_t i = 0; i < rdim; i++) {
        out_file << "result_" << (int)i << ", ";
    }
    out_file << "elapsed_secs" << std::endl;

    double T = TIME; // Original time horizon
    double t_target = 0.5; // Target time t = 1/2
    uint16_t d[8] = {1,2,5,10,20,50,100,200};

    for (uint16_t j = 0; j < sizeof(d) / sizeof(d[0]); j++) {
        for (uint8_t n = N_MAX; n <= N_MAX; n++) {
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            ArrayXd xi = initial_value; // x(0) = (0, 0, ..., 0)
            ArrayXd result = ml_picard(n, n, d[j], xi, 0., t_target, true); // Evaluate at t = 1/2

            std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
            double elapsed_secs = double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000. / 1000.;
            std::cout << "t: " << t_target << std::endl << "d: " << (int)d[j] << std::endl;
            std::cout << "n: " << (int)n << std::endl << "Result:" << std::endl << result << std::endl;
            std::cout << "Elapsed secs: " << elapsed_secs << std::endl << std::endl;

            out_file << (int)d[j] << ", " << t_target << ", " << (int)n << ", ";
            for (uint8_t i = 0; i < rdim; i++) {
                out_file << result(i) << ", ";
            }
            out_file << elapsed_secs << std::endl;
        }
    }

    out_file.close();

    return 0;
}

ArrayXd f(const ArrayXd &v) {
    fn(v);
    return ret;
}

void mlp_thread(mlp_t &mlp_args) {
    mlp_args.res = mlp_call(mlp_args.m, mlp_args.n, mlp_args.l, mlp_args.d, mlp_args.x, mlp_args.s, mlp_args.t);
}

ArrayXd mlp_call(uint8_t m, uint8_t n, uint8_t l, uint16_t d, ArrayXd &x, double s, double t) {
    ArrayXd a = ArrayXd::Zero(rdim, 1);
    ArrayXd b = ArrayXd::Zero(rdim, 1);
    double r = 0.;
    ArrayXd x2;
    uint32_t num;
    static thread_local std::mt19937 generator(128 + clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    static thread_local std::normal_distribution<> normal_distribution{0., 1.};
    static thread_local std::uniform_real_distribution<double> uniform_distribution(0., 1.);
    if (l < 2) {
        num = (uint32_t)(pow(m, n - l) + 0.5);
        for (uint32_t k = 0; k < num; k++) {
            r = s + (t - s) * uniform_distribution(generator);
            x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
            x2 = X_sde(s, r, x, x2);
            b += f(ml_picard(m, l, d, x2, r, t, false));
        }
        a += (t - s) * (b / ((double)num));
    } else {
        num = (uint32_t)(pow(m, n - l) + 0.5);
        for (uint32_t k = 0; k < num; k++) {
            r = s + (t - s) * uniform_distribution(generator);
            x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
            x2 = X_sde(s, r, x, x2);
            b += (f(ml_picard(m, l, d, x2, r, t, l > m - 5)) - f(ml_picard(m, l - 1, d, x2, r, t, l > m - 5)));
        }
        a += (t - s) * (b / ((double)num));
    }
    return a;
}

ArrayXd ml_picard(uint8_t m, uint8_t n, uint16_t d, ArrayXd &x, double s, double t, bool start_threads) {
    if (n == 0) return ArrayXd::Zero(rdim, 1);

    ArrayXd a = ArrayXd::Zero(rdim, 1);
    ArrayXd a2 = ArrayXd::Zero(rdim);
    ArrayXd b = ArrayXd::Zero(rdim, 1);

    double r = 0.;
    std::thread threads[16];
    mlp_t mlp_args[16];
    ArrayXd x2;
    uint32_t num;
    static thread_local std::mt19937 generator(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    static thread_local std::normal_distribution<> normal_distribution{0., 1.};
    static thread_local std::uniform_real_distribution<double> uniform_distribution(0., 1.);

    if (start_threads) {
        for (uint8_t l = 0; l < n; l++) {
            mlp_t mlp_arg;
            mlp_arg.m = m;
            mlp_arg.n = n;
            mlp_arg.l = l;
            mlp_arg.d = d;
            mlp_arg.x = x.replicate(1, 1);
            mlp_arg.s = s;
            mlp_arg.t = t;
            mlp_arg.res = 0.;
            mlp_args[l] = mlp_arg;
            threads[l] = std::thread(mlp_thread, std::ref(mlp_args[l]));
        }

        num = (uint32_t)(pow(m, n) + 0.5);
        for (uint32_t k = 0; k < num; k++) {
            x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
            x2 = X_sde(s, t, x, x2);
            g(x2);
            a2 += tmp;
        }

        a2 /= (double)num;

        for (uint8_t l = 0; l < n; l++) {
            threads[l].join();
            a += mlp_args[l].res;
        }
    } else {
        for (uint8_t l = 0; l < std::min(n, (uint8_t)2); l++) {
            b = ArrayXd::Zero(rdim, 1);
            num = (uint32_t)(pow(m, n - l) + 0.5);
            for (uint32_t k = 0; k < num; k++) {
                r = s + (t - s) * uniform_distribution(generator);
                x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
                x2 = X_sde(s, r, x, x2);
                b += f(ml_picard(m, l, d, x2, r, t, false));
            }
            a += (t - s) * (b / ((double)num));
        }

        for (uint8_t l = 2; l < n; l++) {
            b = ArrayXd::Zero(rdim, 1);
            num = (uint32_t)(pow(m, n - l) + 0.5);
            for (uint32_t k = 0; k < num; k++) {
                r = s + (t - s) * uniform_distribution(generator);
                x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
                x2 = X_sde(s, r, x, x2);
                b += (f(ml_picard(m, l, d, x2, r, t, false)) - f(ml_picard(m, l - 1, d, x2, r, t, false)));
            }
            a += (t - s) * (b / ((double)num));
        }

        num = (uint32_t)(pow(m, n) + 0.5);
        for (uint32_t k = 0; k < num; k++) {
            x2 = ArrayXd::NullaryExpr(d, [&](){ return normal_distribution(generator); });
            x2 = X_sde(s, t, x, x2);
            g(x2);
            a2 += tmp;
        }

        a2 /= (double)num;
    }

    return a + a2;
}