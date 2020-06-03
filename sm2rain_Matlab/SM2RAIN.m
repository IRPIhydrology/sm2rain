function [NS,R,RMSE,PPsim,Psim,Pobs,D]=SM2RAIN(name,PAR,NN,FIG,deltaSM)
if nargin==4,deltaSM=0.005;end
data=load([name,'.txt']);
D=data(1:end-1,1); Pobs=data(1:end-1,3); SM=data(:,2);
Z=PAR(1);
a=PAR(2);
b=PAR(3);

N=length(Pobs);
Psim=zeros(N,1);
for t=2:N
    if (SM(t)-SM(t-1))>deltaSM
%         SMav=(SM(t)+SM(t-1))./2;
%         Psim(t-1)=Z.*(SM(t)-SM(t-1))+a.*SMav.^b;
        Psim(t-1)=Z.*(SM(t)-SM(t-1))+(a.*SM(t).^b+a.*SM(t-1).^b)./2;
    end
end

Psim(isnan(SM(2:end)))=NaN;
Pobs(isnan(Psim))=NaN;
PPsim=Psim;
% Temporal aggregation
if NN>1
    MM=length(Psim);
    L=floor(MM/NN);
    Psim=reshape(Psim(1:L*NN),NN,L);
    Psim=sum(Psim)';
    Pobs=reshape(Pobs(1:L*NN),NN,L);
    Pobs=sum(Pobs)';
    SM=reshape(SM(1:L*NN),NN,L);
    SM=mean(SM)';
    D=reshape(D(1:L*NN),NN,L);
    D=mean(D)';
end

% Calculation of model performance
IDcomp=(Pobs>-1);
RMSE=nanmean((Psim(IDcomp)-Pobs(IDcomp)).^2).^0.5;
NS=1-nansum((Psim(IDcomp)-Pobs(IDcomp)).^2)./nansum((Pobs(IDcomp)-nanmean(Pobs(IDcomp))).^2);
R=corr(Psim(IDcomp),Pobs(IDcomp),'rows','complete');

% Figure
if FIG==1
    close all
    set(gcf,'paperpositionmode','manual','paperposition',[1 1 16 10])
    set(gcf,'position',[100   100   640   400])

    axes('Position',[0.1 0.3 0.8 0.60]);
    set(gca,'Fontsize',12)
    s=(['NS= ',num2str(NS,'%4.3f'),...
        ' R= ',num2str(R,'%4.3f'),...
        ' RMSE= ',num2str(RMSE,'%4.3f'),' mm']);
    title(['\bf',s]);
    hold on
    plot(D,Pobs,'g','Linewidth',3)
    plot(D,Psim,'r--','Linewidth',2);
    grid on, box on
    ylabel('rain [mm]')
    hh=legend('P_o_b_s','P_s_i_m');
    set(hh,'Fontsize',10)
    datetick('x',12)
    axis([data(1,1)-1 data(end,1)+1 min(Pobs)+0.05 max(Pobs)+0.05])
    set(gca,'Xticklabel','')

    axes('Position',[0.1 0.1 0.8 0.20]);
    set(gca,'Fontsize',13)
    hold on
    if NN==1,plot(D,SM(1:end-1),'color',[.5 .5 .5],'Linewidth',3);end
    if NN>1,plot(D,SM(1:length(D)),'color',[.5 .5 .5],'Linewidth',3);end
    grid on, box on
    ylabel('soil saturation [-]')
    datetick('x',12)
    axis([data(1,1)-1 data(end,1)+1 -0.05 1.1])
    s=num2str(NN);
    if NN<10,s=['0',num2str(NN)];end

    print(gcf,['SM2R_',name,'_',s],'-dpng','-r250')
end
