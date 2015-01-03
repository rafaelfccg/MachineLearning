

load flu
y = double(flu(:,2:end-1));  % response = regional queries
x = flu.WtdILI;              % predictor = national CDC estimates
[nobs,nregions] = size(y);

% Create and fit model with separate intercepts but common slope
X = cell(nobs,1);
for j=1:nobs
  X{j} = [eye(nregions), repmat(x(j),nregions,1)];
end
[b,sig,resid,vars,loglik] = mvregress(X,y);

% Plot raw data with fitted lines
B = [b(1:nregions)';repmat(b(end),1,nregions)]
subplot(3,1,1);
xx = linspace(.5,3.5)';
h = plot(x,y,'x', xx, [ones(size(xx)),xx]*B,'-');
for j=1:nregions; set(h(nregions+j),'color',get(h(j),'color')); end
regions = flu.Properties.VarNames;
legend(regions{2:end-1},'location','NW')

% Create and fit model with separate intercepts and slopes
for j=1:nobs
  X{j} = [eye(nregions), x(j)*eye(nregions)];
end
[b,sig,resid,vars,loglik2] = mvregress(X,y);

% Plot raw data with fitted lines
B = [b(1:nregions)';b(nregions+1:end)']
subplot(3,1,2);
h = plot(x,y,'x', xx, [ones(size(xx)),xx]*B,'-');
for j=1:nregions; set(h(nregions+j),'color',get(h(j),'color')); end

% Likelihood ratio test for significant difference
chisq = 2*(loglik2-loglik)
p = 1-chi2cdf(chisq, nregions-1)

% Create and fit model with separate intercepts and slopes in matrix form
X = [ones(size(x)),x];
[b,sig,resid,vars,loglik2] = mvregress(X,y);

% Plot raw data with fitted lines
B = b
subplot(3,1,3);
h = plot(x,y,'x', xx, [ones(size(xx)),xx]*B,'-');
for j=1:nregions; set(h(nregions+j),'color',get(h(j),'color')); 
end