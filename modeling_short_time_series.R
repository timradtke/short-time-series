# This script accompanies the "Modeling Short Time Series with Prior Knowledge"
# blog post which you can find at 
# https://minimizeregret.com/short-time-series-prior-knowledge/

# The script will load the data, fit models from the `forecast` and `prophet`
# packages, as well as the final models implemented in Stan.
# The Stan models are defined in the corresponding .stan files.
# You will see how you can perform prior/posterior predictive checks with your
# fitted RStan model objects.

################################################################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)

library(forecast) # http://pkg.robjhyndman.com/forecast/
library(prophet)  # https://facebook.github.io/prophet/

# see https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started for
# installation instructions for Stan and RStan
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

################################################################################

# We load the full data set for Citi Bike station with ID 360 in New York City
# The original raw data is publicly available at
# https://s3.amazonaws.com/tripdata/index.html

citi <- readRDS("citi_bike_360.Rds")

# We want to forecast based on only the first three months of data
citi_train <- filter(citi, date < as.Date("2013-10-15"))

# take a first look
ggplot(citi_train, aes(x = date, y = rides)) +
  geom_point() +
  geom_line() +
  labs(x = "Date", y = "Sales") +
  theme_bw() + 
  coord_cartesian(ylim = c(0,max(citi_train$rides)))

################################################################################

# First off, we try what happens when we throw the automated ARIMA procedure
# at the time series. See the `forecast` package for more information.
# http://pkg.robjhyndman.com/forecast/

# forecast::auto.arima() only accepts ts() objects; use weekly seasonality
rides <- ts(citi_train$rides, frequency = 7)
arima_fit <- forecast::auto.arima(rides)
arima_fc <- forecast::forecast(arima_fit, h = 360)

# The straight line forecast shown in the blog post
autoplot(arima_fc) +
  theme_bw() +
  labs(x = "Date", y = "Sales", title = "") +
  theme(legend.position = "none")

################################################################################

# Before we use Fourier terms in our models, let's shortly explore how they work
# https://en.wikipedia.org/wiki/Fourier_series

# You can use Fourier terms to approximate arbitrary functions. We now first
# try to approximate a step function with period length 360.

step_function <- rep(rep(c(-1, 1), each = 180), each = 4)
fouriers <- as.data.frame(
  forecast::fourier(ts(1:(4*360), frequency = 360), K = 6))

fouriers %>%
  tidyr::gather(Order, Value, -Date) %>%
  mutate(Order2 = substr(Order, 2, 2),
         Order1 = substr(Order, 1, 1)) %>%
  ggplot() +
  geom_line(aes(Date, Value, group = Order, color = Order1)) +
  facet_grid(Order2~.) +
  theme_bw() +
  theme(legend.position = "none")

################################################################################

# Add Fourier terms first for the weekly seasonality to the ARIMA model

xreg_fourier <- forecast::fourier(rides, K = 3)
xreg_fourier_future <- forecast::fourier(rides, K = 3, h = 360)
arima_fit <- forecast::auto.arima(rides, seasonal = FALSE, xreg = xreg_fourier)
arima_fc <- forecast::forecast(arima_fit, h = 360, xreg = xreg_fourier_future)

autoplot(arima_fc) +
  theme_bw() +
  labs(x = "Date", y = "Sales", title = "") +
  theme(legend.position = "none")

################################################################################

# Add Fourier terms for both weekly and yearly seasonality to the ARIMA model

xreg_fourier <- cbind(forecast::fourier(rides, K = 3),
                      forecast::fourier(ts(rides, frequency = 365.25), K = 6))
xreg_fourier_future <- cbind(forecast::fourier(rides, K = 3, h = 360),
                             forecast::fourier(ts(rides, frequency = 365.25), 
                                               h = 360, K = 6))
arima_fit <- forecast::auto.arima(rides, seasonal = FALSE, xreg = xreg_fourier)
arima_fc <- forecast::forecast(arima_fit, h = 360, xreg = xreg_fourier_future)

# We overfit extremely due to the degrees of freedom we get from the Fouriers

autoplot(arima_fc) +
  theme_bw() +
  labs(x = "Date", y = "Sales", title = "") +
  theme(legend.position = "none")

################################################################################

# Now try to fit a `prophet` model with yearly seasonality;
# Note that `prophet` also uses Fourier terms, as well as Bayesian inference
# via MCMC and Stan in the background. So it's quite comparable to what we do
# here.

# Bring the data frame into the required format
pp <- citi_train %>%
  select(date, rides) %>%
  rename(ds = date, y = rides)

m <- prophet(pp, n.changepoints = 1, changepoint.range = 0.5,
             yearly.seasonality = 8, weekly.seasonality = 3,
             mcmc.samples = 4000, uncertainty.samples = 4000,
             cores = 4)
pf <- make_future_dataframe(m, periods = 180)
pfc <- predict(m, pf)

plot(m, pfc) +
  theme_bw()

################################################################################

# In the following come the parts for the Bayesian Time Series Model
# Again, see https://minimizeregret.com/short-time-series-prior-knowledge/
# for the detailed description of what we're doing.

################################################################################

# Plotting the Normal prior on the trend regressor

df <- data.frame(x = seq(-0.025,0.075,length.out=1000),
                 y = dnorm(seq(-0.025,0.075,length.out=1000),0.03,0.02))

ggplot(df, aes(x = x, y = y)) +
  geom_line() +
  labs(x = "Growth", y = "Density") +
  theme_bw()

# How large is the probability of drawing a negative trend from this 
# distribution?

round(pnorm(0, 0.03, 0.02, lower.tail = TRUE)*100, 2)

################################################################################

# Download the temperature data from Kaggle (sorry).
# https://www.kaggle.com/selfishgene/historical-hourly-weather-data#temperature.csv

# then read it in
temperature <- readr::read_csv("temperature.csv")

# summarize the data set to daily observations for New York City
daily_temp <- temperature %>%
  select(datetime, `New York`) %>%
  rename(temperature = `New York`) %>%
  mutate(date = as.Date(datetime)) %>%
  group_by(date) %>%
  summarize(max_temp = round(max(temperature, na.rm = TRUE))) %>%
  filter(!is.infinite(max_temp)) %>%
  filter(date < as.Date("2015-10-01")) # take 3 years

ggplot(daily_temp) +
  geom_line(aes(x = date, y = max_temp)) +
  theme_bw() +
  labs(x = "Date", y = "Temperature [Kelvin]")

################################################################################

# The next step for us is to fit a simple regression with Fourier terms
# for yearly seasonality on the temperature data to get posterior distributions
# on the Fourier terms' coefficients which we can subsequently employ in our
# actual model for the rides.

# In Stan, we fit the following model:
pois_stan <- stan_model("pois_reg.stan")
print(pois_stan)

# Before we can fit the model, we need to prepare the data objects that are
# defined in pois_stan and passed to Stan via rstan::sampling().

# We start by defining a very long date vector which we will use for the
# rides data as well, and to make predictions afterwards.
# Note that it is important to align the Fourier terms we use for the 
# temperature and the rides data. Else we will model a peak in summer, and then
# it lands in winter in the second model because we misaligned the Fourier terms
# This is what the dates help us with.

all_dates <- seq(as.Date("2012-01-01"), as.Date("2020-01-01"), by = 1)
fourier_yearly <- forecast::fourier(ts(all_dates, frequency = 365.25), K = 6)
fourier_table <- data.frame(date = all_dates, fourier_yearly)
names(fourier_table)[2:13] <- paste0("fourier", 1:12)
fourier_train <- filter(fourier_table, date %in% daily_temp$date)

# Because the coefficients are interpreted as % changes in the log-linear model,
# We bring the temperature data closer to the scale of the rides data.
# A change from 1 to 10 is very different than a change from 91 to 100.

y_t <- daily_temp$max_temp - min(daily_temp$max_temp)
xreg <- as.matrix(fourier_train[,paste0("fourier", 1:12)])
K <- dim(xreg)[2]
N <- length(y_t)

# Fit the first Stan model:

temperature_fit <- sampling(pois_stan,
                            data = list(y = y_t, x = xreg, N = N, K = K),
                            iter = 4000, algorithm = "NUTS", seed = 512)

# The temperature_fit object contains the posterior samples for every
# coefficient. We supplied K sine and cosine regressors, thus we need to extract
# the posterior samples for the corresponding K coefficients.
# We can do so as follows, and then use the bayesplot package to take a look
# at the posterior distributions for each of them.

beta_samples <- as.data.frame(temperature_fit)[,paste0("beta[", 1:K, "]")]
names(beta_samples) <- paste0("Fourier ", 1:12)
bayesplot::mcmc_areas(beta_samples) + 
  theme_bw() +
  labs(y = "Coefficient", x = "Posterior Value")

# We compress the information contained in the posterior samples into Normal
# distributions by extracting the sample mean and standard deviation of each 
# of the posterior samples (we get K means and K standard deviations).
beta_means <- as.numeric(colMeans(beta_samples))
beta_sds <- as.numeric(apply(beta_samples, 2, sd))

# Using these posterior coefficients, we can also look at the implied
# posterior linear combination of the Fourier terms; that is, how did the
# amplitude of each sine and cosine curve change individually,
# and what does the signal look like that they model as linear combination.

# multiply every sine and cosine curve by its posterior mean coefficient
xx <- matrix(rep(beta_means, N), byrow = TRUE, ncol = 12) * xreg
fouriers <- data.frame(xx)
fouriers$date <- daily_temp$date
fouriers$fourier <- rowSums(xx) # their posterior mean linear combination

fouriers %>%
  tidyr::gather(fourier, value, -date) %>%
  mutate(kind = ifelse(fourier == "fourier", 
                       "Linear Combination", "Individual Terms")) %>%
  ggplot(aes(x = date, y = value, group = fourier, color = fourier)) +
  geom_line() +
  facet_grid(kind~.) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(x = "Date", y = "Value")

################################################################################

# We have extracted the posterior Fourier coefficients from the temperature and
# can now transfer them as prior distributions parameterized by mean and
# standard deviation to the actual model of bike sales.

# First, we load the Stan model.

ddnbsr <- stan_model("damped_dynamic_negbin_seasonal_reg_prior.stan")

# Next, we prepare the data.

fourier_citi <- fourier_table %>%
  filter(date %in% citi_train$date) %>%
  select(-date) %>%
  as.matrix()
trend_citi <- matrix((1:length(citi_train$date))/365.25, ncol = 1)
xreg_citi <- as.matrix(citi_train[,paste0("wday", 1:6)])
y_citi <- citi_train$rides
N_citi <- length(y_citi)
K_citi <- dim(xreg_citi)[2]

# Finally, we fit the model and also provide priors as data.

citi_temp_fit <- sampling(object = ddnbsr, 
                          data = list(y = y_citi, 
                                      x = xreg_citi,
                                      fouriers = fourier_citi,
                                      trend = trend_citi,
                                      trend_loc = 0.03,
                                      trend_sd = 0.02,
                                      fourier_loc = beta_means,
                                      fourier_sd = beta_sds,
                                      N = N_citi, K = K_citi,
                                      F = 12),
                          iter = 4000, algorithm = "NUTS", seed = 512)

################################################################################

# Lastly, we use the fitted model to produce forecasts by extracting the
# posterior samples to produce sample paths of potential future outcomes

# Extract the posterior samples of all chains
post_samples <- as.data.frame(citi_temp_fit)
beta_samples <- as.data.frame(citi_temp_fit)[,paste0("beta[", 1:K_citi, "]")]
beta_fourier_samples <- as.data.frame(citi_temp_fit)[,paste0("beta_fourier[", 1:12, "]")]
beta_0_samples <- as.data.frame(citi_temp_fit)[,paste0("beta_0"),drop=FALSE]
beta_trend_samples <- as.data.frame(citi_temp_fit)[,paste0("beta_trend[1]"),drop=FALSE]
y_hats <- colMeans(as.data.frame(citi_temp_fit)[,paste0("y_hat[", 1:N_citi, "]")])

# Load the actual test data
citi_test <- filter(citi, date >= as.Date("2013-10-15"),
                    date <= as.Date("2015-06-30"))

fourier_citi_test <- fourier_table %>%
  filter(date %in% citi_test$date) %>%
  select(-date) %>%
  as.matrix()
trend_citi_test <- matrix((N_citi:(N_citi+length(citi_test$date) - 1))/365.25, ncol = 1)
xreg_citi_test <- as.matrix(citi_test[,paste0("wday", 1:6)])

set.seed(3957)
h <- dim(citi_test)[1]
k <- 8000
mu_fut <- y_fut <- matrix(ncol = k, nrow = h)

# for every posterior sample, compute the seasonal mean model for the future
seasonal <- exp(xreg_citi_test %*% t(beta_samples) + 
                  fourier_citi_test %*% t(beta_fourier_samples) +
                  trend_citi_test %*% t(beta_trend_samples) +
                  matrix(rep(beta_0_samples$beta_0, h), 
                         ncol = k, byrow = TRUE))

# the overall time-varying mean and the outcome have to be computed iteratively
# We start with the first future observation as it's based on the training mean
mu_fut[1,] <- (1 - post_samples$phi - post_samples$alpha) * 
  seasonal[1,] +
  post_samples$phi * post_samples$`mu_t[106]` + 
  post_samples$alpha * y_citi[length(y_citi)]
y_fut[1,] <- rnbinom(k, mu = as.vector(mu_fut[1,]), 
                     size = post_samples$delta)

# and now we do the rest based on the previous prediction
for(i in 2:h) {
  mu_fut[i,] <- (1 - post_samples$phi - post_samples$alpha) * 
    seasonal[i,] +
    post_samples$phi * mu_fut[i-1,] + 
    post_samples$alpha * y_fut[i-1,]
  y_fut[i,] <- rnbinom(k, mu = as.vector(mu_fut[i,]), 
                       size = post_samples$delta)
}

# We now summarize the large sample data frames into a format that is easier to
# plot
df_to_plot <- data.frame(date = c(citi_train$date, citi_test$date),
                         y = c(citi_train$rides, citi_test$rides),
                         yhat_mean = c(rep(NA, length(citi_train$date)),
                                       rowMeans(y_fut)),
                         yhat_q001 = c(rep(NA, length(citi_train$date)),
                                       apply(y_fut, 1, quantile, 0.001)),
                         yhat_q05 = c(rep(NA, length(citi_train$date)),
                                      apply(y_fut, 1, quantile, 0.05)),
                         yhat_q25 = c(rep(NA, length(citi_train$date)),
                                      apply(y_fut, 1, quantile, 0.25)),
                         yhat_q75 = c(rep(NA, length(citi_train$date)),
                                      apply(y_fut, 1, quantile, 0.75)),
                         yhat_q95 = c(rep(NA, length(citi_train$date)),
                                      apply(y_fut, 1, quantile, 0.95)),
                         yhat_q999 = c(rep(NA, length(citi_train$date)),
                                       apply(y_fut, 1, quantile, 0.999)),
                         yhat_example = c(rep(NA, length(citi_train$date)),
                                          y_fut[,512]))

df_past <- df_to_plot[1:dim(citi_train)[1],]
df_future <- df_to_plot[(1+dim(citi_train)[1]):dim(df_to_plot)[1],] %>%
  filter(date <= as.Date("2015-06-30"))

ggplot() +
  geom_vline(aes(xintercept = max(df_past$date)), linetype = 2) +
  geom_ribbon(aes(x = date, ymin = yhat_q05, ymax = yhat_q95), fill = "#cdd2fa", data = df_future) +
  geom_ribbon(aes(x = date, ymin = yhat_q25, ymax = yhat_q75), fill = "#aab0ed", data = df_future) +
  geom_point(aes(x = date, y = y), stroke = 0, data = df_past) +
  geom_point(aes(x = date, y = y), alpha = 0.4, stroke = 0, data = df_future) +
  theme_bw() +
  labs(x = "Date", y = "Sales")
