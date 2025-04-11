clc;
clear;

function [U, S, V] = T_SVD(A)
  A_aux = fft(A, [], 3);
  dimensions = size(A);
  mid = floor((dimensions(3) + 1) / 2);  % Usar floor para asegurar que mid sea un entero
  U_aux = zeros(dimensions);
  S_aux = zeros(dimensions);
  V_aux = zeros(dimensions);
  for i = 1:mid
    [U_aux(:, :, i), S_aux(:, :, i), V_aux(:, :, i)] = svd(A_aux(:, :, i));
  end
  for i = mid + 1:dimensions(3)
    aux = dimensions(3) - i + 2;
    U_aux(:, :, i) = conj(U_aux(:, :, aux));
    S_aux(:, :, i) = S_aux(:, :, aux);
    V_aux(:, :, i) = conj(V_aux(:, :, aux));
  end
  U = ifft(U_aux, [], 3);
  S = ifft(S_aux, [], 3);
  V = ifft(V_aux, [], 3);
  disp(V);
end


function r = t_product(t_1, t_2)
  t_1_aux = fft(t_1,[],3);
  t_2_aux = fft(t_2,[],3);
  dim_1 = size(t_1);
  dim_2 = size(t_2);
  r_aux = zeros(dim_1(1),dim_2(2), dim_1(3));
  mid = floor((dim_1(3)+1)/2);
  for i=1:mid
    r_aux(:,:,i) = t_1_aux(:,:,i)*t_2_aux(:,:,i);
  endfor
  for i=mid+1:dim_1(3)
    aux = dim_1(3) - i + 2;
    r_aux(:,:,i) = conj(r_aux(:,:,aux));
  endfor
  r = ifft(r_aux,[],3);
end
data = [1 2 3 4 5 6 7 8 9 ...
        10 20 30 40 50 60 70 80 90 ...
        100 200 300 400 500 600 700 800 900];

T = reshape(data, [3, 3, 3]);
T(:,:,1)= T(:,:,1)';
T(:,:,2)= T(:,:,2)';
T(:,:,3)= T(:,:,3)';
[U,S,V] = T_SVD(T);
V_perm = permute(V, [2, 1, 3]);
l = t_product(U,S);
r = t_product(l,V_perm);


