data{
  int <lower = 0> n;
  int <lower = 0> p;
  matrix[n,p] X ;
  real y [n];
  real <lower =0> sigma_y;
}
parameters{
  vector[p] beta;
}
model{
  y ~ normal(X * beta, sigma_y);
}
