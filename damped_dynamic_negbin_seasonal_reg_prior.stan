data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> F;
  int y[N];
  matrix[N, K] x;
  matrix[N, F] fouriers;
  matrix[N, 1] trend;
  real trend_loc;
  real trend_sd;
  real fourier_loc[F];
  real fourier_sd[F];
}
parameters {
  real<lower=0,upper=1> phi;
  real<lower=0,upper=1-phi> alpha;
  real delta_inv;
  vector[K] beta;
  vector[F] beta_fourier;
  vector[1] beta_trend;
  real beta_0;
}
transformed parameters {
  real<lower=0> delta;
  real<lower=0> mu_t[N];
  vector<lower=0>[N] seasonal;
  delta = 1/pow(delta_inv,2);
  seasonal = exp(x * beta + fouriers * beta_fourier + 
                  trend * beta_trend + beta_0);
  mu_t[1] = y[1];
  for (n in 2:N) {
    mu_t[n] = (1 - phi - alpha) * seasonal[n] + 
                phi * mu_t[n-1] + alpha * y[n-1];
  }
}
model {
  phi ~ gamma(1,10);
  alpha ~ gamma(0.5,10);
  delta_inv ~ normal(0,0.5);
  beta_fourier ~ normal(fourier_loc, fourier_sd);
  beta_trend ~ normal(trend_loc, trend_sd);
  beta ~ normal(0,0.5);
  beta_0 ~ normal(2,1);
  for (n in 2:N) {
    target += neg_binomial_2_lpmf(y[n] | mu_t[n], delta);
  }
}
generated quantities {
  int y_hat[N];
  
  for (n in 1:N)
    y_hat[n] = neg_binomial_2_rng(mu_t[n], delta);
}
