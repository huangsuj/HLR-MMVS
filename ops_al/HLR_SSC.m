%% Compute semi-supervised classification performance

%%%%%%%%%%%%% Input %%%%%%%%%%%%%%%%%%%%%%
% X: multi-view data matrix (each view with nFeature x nSample)
% gt: class label column

%%%%%%%%%%%% Output %%%%%%%%%%%%%%%%%%%%%%
% ACC: classification performance

function [results] = HLR_SSC(X, gt, ratio)
%% For convinience, we assume the order of the tensor is always 3;

%addpath('tSVD','proxFunctions','solvers','twist','data-ORL');
% load('./multiview_dataset/my_ORL.mat'); % bestComb:X3,X2,X1 lambda = 100

class_num = length(unique(gt));
%% Note: each column is an sample (same as in LRR)

 for v=1:length(X)
     X{v} = double(X{v});
    [X{v}]=NormalizeData(X{v});
%     X{v} = X{v}';
end
% Initialize...

N = size(X{1},2); %sample number
labeled_N = floor(N*ratio);
% % Random select the labeled data
% each_class_num = floor(N/class_num);  
% part = floor(ratio*each_class_num);
% labeled_N = part*class_num; 
% list = sort(randperm(each_class_num,part));  
% List = [];
% for c = 1:class_num
%     List = [List list+(c-1)*each_class_num];
% end
% List_ = setdiff(1:1:N,List); % the No. of unlabeled data
% rowrank = [List List_];
rowrank = randperm(size(gt, 1)); % 随机打乱的数字，从1~行数打乱
gt = gt(rowrank, :);
% gt = gt+1;
for v=1:length(X)
    X{v} = X{v}(:, rowrank);
end
K = length(X); 

for k=1:K
    Z{k} = zeros(N,N); %Z{2} = zeros(N,N);
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    B{k} = zeros(N,N);
    Q{k} = zeros(N,N);
    L{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); %E{2} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); %Y{2} = zeros(size(X{k},1),N);
    
%---------------------------modify-----------------------------
    gamma{k} = 1/K;                                     
end                  
%     F_l = zeros(semi_num,cls_num);              
    F_u = zeros(N-labeled_N,class_num);            
    Y_l = zeros(labeled_N,class_num);          

for i=1:(labeled_N)                                          
    Y_l(i,gt(i))=1;                                 
end                                                                                    
                                                                  
%---------------------------modify-----------------------------


%w = zeros(N*N*K,1);
%g = zeros(N*N*K,1);

dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX = [N, N, K];
%set Default
parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;


Isconverg = 0;epson = 1e-5;
lambda1 = 0.2; %0.2 best
lambda2 = 0.4; %0.4 best

% lambda1 = 0.1; %1.5 best
% lambda2 = 1.2;
% lambda3 = 0.4;

iter = 0;
mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 2;
mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 2;
rho = 10e-5; max_rho = 10e10; pho_rho = 2;
tic;
start = 1;
for k=1:K
    tmp_inv{k} = inv(2*eye(N,N)+X{k}'*X{k});
end
while(Isconverg == 0)
    fprintf('----processing iter %d--------\n', iter+1);
    %-------------------0 update L^k-------------------------------
    for k=1:K
        if start==1
          Weight{k} = constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, 3);
          Diag_tmp = diag(sum(Weight{k}));
          L{k} = Diag_tmp - Weight{k};
        else
        %------------modified to hyper-graph---------------
          P =  (abs(Z{k})+abs(Z{k}'))./2;
          param.k = 3;
          HG = gsp_nn_hypergraph(P', param);
          L{k} = HG.L;
        end
        
%         Weight{k} = constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, 10);
%         Diag_tmp = diag(sum(Weight{k}));
%         L{k} = Diag_tmp - Weight{k};
    end
    start = 0;
  
    for k=1:K
        %-------------------1 update Z^k-------------------------------    
        %Z{k}=inv(2*eye(N,N)+X{k}'*X{k}) * ( (X{k}'*Y{k} + B{k} + \mu2*Q{k} - W{k} + \rho*G{k})/\mu1
        %                                    + X{k}'*X{k} - X{k}'*E{k});
        tmp = (X{k}'*Y{k} + B{k} + mu2*Q{k} - W{k} + rho*G{k})/mu1 + X{k}'*X{k} - X{k}'*E{k};
        Z{k}=tmp_inv{k}*tmp;
    end
    
    %-------------------2 update E^k-------------------------------
    F = [];
    for k=1:K    
        tmp = X{k}-X{k}*Z{k}+Y{k}/mu1;
        F = [F;tmp];
    end
    %F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu];
    [Econcat] = solve_l1l2(F,lambda1/mu1);
    %[Econcat,info] = prox_l21(F, 0.5/1);
    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end
  
    %-------------------3 update Q^k-------------------------------
    for k=1:K
        Q{k} = (mu2*Z{k} - B{k})*inv(2*lambda2*L{k} + mu2*eye(N,N));
    end
    
    %-------------------4 update G---------------------------------
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:); 
    %twist-version
    [g, objV] = wshrinkObj(z + 1/rho*w,1/rho,sX,0,3)   ;
%    [g, objV] = shrinkObj(z + (1/rho)*w,...
%                         1/rho,myNorm,sX,parOP);
    G_tensor = reshape(g, sX);
    
    %-------------------6 update auxiliary variable---------------
    %k循环没写
    w = w + rho*(z - g);
    W_tensor = reshape(w, sX);
    for k=1:K
        Y{k} = Y{k} + mu1*(X{k}-X{k}*Z{k}-E{k});
        B{k} = B{k} + mu2*(Q{k}-Z{k});
    
        G{k} = G_tensor(:,:,k);
        W{k} = W_tensor(:,:,k);
    end
    
    %record the iteration information
    history.objval(iter+1)   =  objV;

    %% coverge condition
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
%             fprintf('    norm_Z %7.10f    ', history.norm_Z);
            Isconverg = 0;
        end
        
        if (norm(Z{k}-G{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-G{k},inf);
%             fprintf('norm_Z_G %7.10f    \n', history.norm_Z_G);
            Isconverg = 0;
        end
    end
   
    if (iter>20)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu1 = min(mu1*pho_mu1, max_mu1);
    mu2 = min(mu2*pho_mu2, max_mu2);
    rho = min(rho*pho_rho, max_rho);
end
% S1 = 0;
% for k=1:K
%     S1 = S1 + abs(Z{k})+abs(Z{k}');
% end

%---------------------------modify-----------------------------
for k=1:K
      Weight{k} = constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, 9);
      Diag_tmp = diag(sum(Weight{k}));
%       L{k} = Diag_tmp - Weight{k};
      temp_ = diag(1./sqrt( diag(Diag_tmp)+1e-8));
      L{k} = eye(N)-temp_*Weight{k}*temp_;
end

% for k=1:K
%     P =  (abs(Z{k})+abs(Z{k}'))./2;
%     param.k = 3;
%     HG = gsp_nn_hypergraph(P', param);
%     L{k} = full(HG.L);
% end
thresh = 10e-7;          
maxIter = 20;
for iter = 1:maxIter
    %-------------------1 update L_star--------
    L_sum = zeros(N);
    for k=1:K
        L_sum = L_sum + gamma{k}*L{k};
    end
    %-------------------2 update Fu------------
    L_ul = L_sum((labeled_N+1):N,1:labeled_N);
    L_uu = L_sum((labeled_N+1):N,(labeled_N+1):N);
    F_u = -1*inv(L_uu)*L_ul*Y_l;
    %-------------------3 update gamma---------
    F_all = [Y_l;F_u];             
    for k=1:K
        gamma{k}=0.5/sqrt(trace(F_all'*L{k}*F_all));
    end
    %% coverge condition
    % Calculate objective value
    obj = 0;
    for k=1:K
          obj = obj+sqrt(trace(F_all'*L{k}*F_all));
    end
    Obj(iter) = obj;
    if iter>2
        Obj_diff = ( Obj(iter-1)-Obj(iter) )/Obj(iter-1);
        if Obj_diff < thresh
            fprintf('F converge at iter: %d \n', iter);
            break;
        end
    end
end
[~,predict_label]=max(F_u,[],2);
% Y_all = zeros(length(gt), 1);
% Y_req_all = zeros(length(gt), length(unique(gt)));
% label_idx = 1:labeled_N;
% unlabel_id = labeled_N+1:N;
[~,train_lablel] = max(Y_l,[],2);
% all_label = [train_lablel; predict_label];
all_req = [Y_l; F_u];
% Y_all(rowrank,:) =  gt;
% Y_req_all(rowrank,:) =  all_req;
% Y_all(unlabel_id,:) = predict_label;
results = classification_metrics(gt(labeled_N+1:end), predict_label);
save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\Citeseer\HLR_label.mat', 'gt');
save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\Citeseer\HLR_representation.mat', 'all_req');