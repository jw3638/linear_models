bootstrapping
================

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.2     ✔ tibble    3.3.0
    ## ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ## ✔ purrr     1.1.0     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(p8105.datasets)
library(modelr)
```

Simulate 2 datasets

``` r
set.seed(1)
n_samp = 250

sim_df_const =
  tibble(
    x = rnorm(n_samp, 1, 1),
    error = rnorm(n_samp, 0, 1),
    y = 2 + 3 * x + error
  )

sim_df_nonconst =
  sim_df_const |>
  mutate(
    error = .75 * error *x,
    y = 2 + 3 * x + error
  )
```

Look at these data

``` r
sim_df_nonconst |>
  ggplot(aes(x =x, y=y)) +
  geom_point()
```

![](bootstrapping_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

What does `lm` do for these?

``` r
sim_df_const |>
  lm(y ~ x, data = _) |>
  broom::tidy() |>
  knitr::kable(digits = 3)
```

| term        | estimate | std.error | statistic | p.value |
|:------------|---------:|----------:|----------:|--------:|
| (Intercept) |    1.977 |     0.098 |    20.157 |       0 |
| x           |    3.045 |     0.070 |    43.537 |       0 |

Write a function that draws a bootsrtap sample

``` r
boot_sample = function(df) {
  sample_frac(df, size =1, replace = TRUE) # The replace = TRUE: Give me a different sample each time 
  
}
```

Does this work? Get different bootstrap samples every time you run it,
but from the same underlying dataset

``` r
sim_df_nonconst |>
  boot_sample() |>
  ggplot(aes(x = x, y=y)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  xlim(c(-2,4)) +
  ylim(c(-5,16))
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](bootstrapping_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

To formalize this, extract results.

``` r
boot_straps =
  tibble(
    iter = 1:5000
  ) |>
  mutate(
    bootstrap_sample = map(iter, \(i) boot_sample(df = sim_df_nonconst))
)
```

Quick check

``` r
boot_straps |>
  pull(bootstrap_sample) |>
  nth(2) |>
  ggplot(aes(x = x, y=y)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  xlim(c(-2,4)) +
  ylim(c(-5,16))
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](bootstrapping_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

Actually run my analyses!

``` r
bootstrap_results =
  boot_straps |>
  mutate(
    fits = map(bootstrap_sample, \(df) lm(y ~ x, data =df)),
    results = map(fits, broom::tidy)
  )
```

Look at results.

``` r
bootstrap_results |>
  select(iter, results) |>
  unnest(results) |>
  group_by(term) |>
  summarize(
    mean = mean(estimate),
    se = sd(estimate)
  )
```

    ## # A tibble: 2 × 3
    ##   term         mean     se
    ##   <chr>       <dbl>  <dbl>
    ## 1 (Intercept)  1.93 0.0762
    ## 2 x            3.11 0.103

Look at these first

``` r
bootstrap_results |>
  select(iter, results) |>
  unnest(results) |>
  filter(term == "x") |>
  ggplot(aes(x = estimate)) +
  geom_density()
```

![](bootstrapping_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
bootstrap_results |>
  select(iter, results) |>
  unnest(results) |>
  group_by(term) |>
  summarize(
    ci_lower = quantile(estimate, 0.025),
    ci_upper = quantile(estimate, 0.975)
  )
```

    ## # A tibble: 2 × 3
    ##   term        ci_lower ci_upper
    ##   <chr>          <dbl>    <dbl>
    ## 1 (Intercept)     1.78     2.09
    ## 2 x               2.91     3.32

## Do bootstrappinge exaple again but faster his time (more consilidated using modelr function)

``` r
bootstrap_results =
  sim_df_nonconst |>
  bootstrap(n=10) |>
  mutate(
    df = map(strap, as_tibble),
    fits = map(df, \(df) lm(y~x, data = df)),
    results = map(fits, broom::tidy)
  ) |>
  select(.id, results) |>
  unnest(results)
```

``` r
bootstrap_results |>
  group_by(term) |>
  summarize(
    mean = mean(estimate),
    sd = sd(estimate)
  )
```

    ## # A tibble: 2 × 3
    ##   term         mean     sd
    ##   <chr>       <dbl>  <dbl>
    ## 1 (Intercept)  1.91 0.0662
    ## 2 x            3.10 0.0912

If you don’t trust assumptions, run bootstrap and check using Airbnb
dataset

``` r
data("nyc_airbnb")

#data cleaning
nyc_airbnb =
  nyc_airbnb |>
  mutate(stars = review_scores_location / 2) |>
  rename(
    borough = neighbourhood_group
  ) |>
  filter(
    borough != "Staten Island"
  ) |>
  drop_na(price, stars, room_type) |>
  select(price, stars, room_type, borough)
```

What does this look like?

``` r
nyc_airbnb |>
  ggplot(aes(x = stars, y = price, color = room_type)) +
  geom_point(alpha = .5)
```

![](bootstrapping_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

Try to do the bootsrap

``` r
airbnb_bootstrap_results =
  nyc_airbnb |>
  filter(borough == "Manhattan") |>
  bootstrap(n =1000) |>
  mutate(
    df = map(strap, as_tibble),
    fits = map(df, \(df) lm(price ~ stars + room_type, data = df)),
    results = map(fits, broom::tidy)
  ) |>
  select(.id, results) |>
  unnest(results)
```

Look at the distribution of the slope for stars (ratings of airbnbs)

``` r
airbnb_bootstrap_results |>
  filter(term == "stars") |>
  ggplot(aes(x = estimate)) +
  geom_density()
```

![](bootstrapping_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->
