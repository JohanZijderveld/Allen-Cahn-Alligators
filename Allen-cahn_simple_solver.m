%Allen-Cahn Solver

%define constants
eps = 0.01; a = -2; b = 4; T = 10;
h = 0.01; tau = 0.01;
x = (a:h:b); N = length (x);
ga = 0; gb = 0;
utime = zeros(N,T/tau+1);
utimeback = zeros (N,T/tau+1);


%initial conditions (random noise)
rng(2);
u = 0.1*rand(N,1);
uback = u; 

%Create the negative Laplacian
%Laplacian matrix
%Construct the Laplacian matrix using finite difference method
one_vec = ones(N,1);
lap = spdiags([1, -2, 1],-1:1,N,N);
lap(1,1) = ga
lap (N,N) = gb
A = (eye(N)-tau*eps^2*lap/h^2)

%Forward Euler
for step = 1:(T/tau+1)
    utime(:,step) = u;
    u = u + tau*(eps^2 * (lap*u) /(h^2) + u - u.^3);
end

%Backwards Euler 
for steps = 1:(T/tau+1)
    utimeback(:,steps) = uback;
    uback = A\(uback + tau * (uback-uback.^3))
end

%for task 3, you can plot this with the things from below 
%utime is an array of the solutions at given times (forward)
%utimeback is the same for backwards
%the plotting can be done below
%also plot the analytical solution from task 2

figure 
plot(x, utime(:,1:150:end), 'linewidth',2); hold on
xlabel('x','FontSize', 12); ylabel('u(x,t)','FontSize', 12);

figure
plot(x, utimeback(:, 1:150:end), '-');
xlabel('x','FontSize', 12); ylabel('u(x,t)','FontSize', 12);

hold on;
