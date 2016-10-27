function adjacency = randomTAN(N, classIdx)

adjacency = zeros(N,N);
adjacency(classIdx, :) = 1;
adjacency(classIdx, classIdx) = 0;

rp = randperm(N-1);
idx = [1:classIdx-1,(classIdx+1):N];

for k=1:N-2
    adjacency(idx(rp(k)), idx(rp(k+1))) = 1;
end

