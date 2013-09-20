function owh_params = weightLearner( owh_params, triplet )
%WEIGHTLEARNER Summary of this function goes here
%   update weight given new similar or dissimilar pairs

code_diff_pos = (triplet.query_code - triplet.pos_code).^2;
code_diff_neg = (triplet.query_code - triplet.neg_code).^2;

hinge_loss = ( double(code_diff_pos - code_diff_neg) ).^2 * owh_params.cur_weights' + 0.5;
if hinge_loss < 0
    return;
end

grad = (owh_params.cur_weights - owh_params.prev_weights) + owh_params.lamda * (double(code_diff_pos - code_diff_neg));

owh_params.prev_weights = owh_params.cur_weights;
owh_params.cur_weights = owh_params.cur_weights .* exp(-owh_params.eta * grad);

owh_params.cur_weights = owh_params.cur_weights ./ sum(owh_params.cur_weights);

if(sum(isnan(owh_params.cur_weights)) > 0)
    disp('Error');
end

end