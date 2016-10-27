function adjacency = randomBNstructure(N, maxParents)

adjacency = zeros(N,N);
rp = randperm(N);

for k = 1:N
    numParents = ceil(rand * (min(maxParents, k-1)+1))-1;
    rp2 = randperm(k-1);
    adjacency(rp(rp2(1:numParents)), rp(k)) = 1;
end
