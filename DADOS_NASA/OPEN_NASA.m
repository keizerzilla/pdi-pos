% load matrizao.prn
% [matrizao]
% matcov = cov(matrizao)
% vetmedia = mean(matrizao) %retorna a media das colunas em uma linha
% avalor = eig(matcov)
% [avetor,D] = eig(matcov)
 
matcov = cov(matrizao);
vetmedia = mean(matrizao);
[avetor,D] = eig(matcov);
avetor1=avetor';
P=flipud(avetor1);
figure
y30=(P)*((matrizao(30,:))'-(vetmedia)');
subplot(1,3,1)
plot(y30)
x30=(P')*y30+(vetmedia)';
subplot(1,3,2)
plot(x30)
subplot(1,3,3)
plot(matrizao(30,:))
figure
y26=(P)*((matrizao(26,:))'-(vetmedia)');
subplot(1,3,1)
plot(y26)
x26=((P)')*y26+(vetmedia)';
subplot(1,3,2)
plot(x26)
subplot(1,3,3)
plot(matrizao(26,:))
figure
y25=(P)*((matrizao(25,:))'-(vetmedia)');
subplot(1,3,1)
plot(y25)
x25=((P)')*y25+(vetmedia)';
subplot(1,3,2)
plot(x25)
subplot(1,3,3)
plot(matrizao(25,:))
