function [X,NS,R,RMSE]=cal_SM2RAIN(name,NN,deltaSM,X_ini)
if nargin<=2,deltaSM=0.004;end
if nargin<=3,X_ini(1)=.1;X_ini(2)=.02;X_ini(3)=.1;end
[RES,FVAL,EXITFLAG,OUTPUT]=fmincon(@calibOK,X_ini',[],[],[],[],...
     zeros(3,1),ones(3,1),[],...
     optimset('Display','iter','MaxIter',100,'MaxFunEvals',500,...
     'TolFun',1E-8,'Largescale','off','Algorithm','active-set'),...
     name,NN,deltaSM);
X=convert_adim(RES);
save(['PAR_',name,'.dat'],'X','-ascii','-double');
% [NS,R,RMSE]=SM2RAIN(name,X,NN,1,deltaSM);

%---------------------------------------------------------------------------------
function [err]=calibOK(X_0,name,NN,deltaSM)

X=convert_adim(X_0);
[NS,R,RMSE]=SM2RAIN(name,X,NN,0,deltaSM);
err=1-NS;
% save X_PAR

%---------------------------------------------------------------------------------
function X=convert_adim(X_0)
LOW=[   1,    0.0,   1]';
UP =[ 200, 1800.0, 150]';
X=LOW+(UP-LOW).*X_0;
