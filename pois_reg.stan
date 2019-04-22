data {
  int<lower=0> N; // how many observations?
    int<lower=1> K; // how many regressors?
      int y[N];
    matrix[N, K] x;
}
parameters {
  vector[K] beta;
  real beta_0;
}
model {
  beta_0 ~ normal(5.6524, 1); // log(285)
  beta ~ normal(0, 0.25);
  y ~ poisson(exp(x * beta + beta_0));
}
