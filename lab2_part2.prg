' %0 - назва workfile
' %1 - розмір вікна КС
' %2 - 1 для побудови графіку ряду + КС, 
'         2 для побудови графіку ряду + ЕКС,
'         0, якщо графік не будувати

load {%0}
!N = {%1}

' обчислення КС та ЕКС
series SMA_series = @movav(data, !N)
series EMA_series
series weights
!a = 2/(!N+1)

for !i = 1 to !N
	weights(!i) = (1-!a)^(!N-!i+1)
next

for !k = !N to @obs(data)
	scalar weighted_sum = 0
	for !i = 1 to !N
		weighted_sum = weighted_sum + weights(!i)*data(!k-!i+1)
	next
	EMA_series(!k) = weighted_sum/@sum(weights)
next

if {%2} = 1 then plot data SMA_series endif
if {%2} = 2 then plot data EMA_series endif

' обчислення лагів
series r
for !s = 1 to @obs(data)
	r(!s) = 0
	for !k = !s+1 to @obs(data)
		r(!s) = r(!s) + (data(!k) - @mean(data))*(data(!k-!s) - @mean(data))
	next
	r(!s) = r(!s)/(@sumsq(data - @mean(data)))
next

' обчислення матриці Ф
matrix (@obs(data), @obs(data)) Phi
Phi(1,1) = r(1)
for !k = 1 to @obs(data)
	if(!k>1) then
		scalar sum_top = 0
		scalar sum_bottom = 0
		for !j = 1 to !k-1
			sum_top = sum_top + Phi(!k-1, !j)*r(!k-!j)
			sum_bottom = sum_bottom + Phi(!k-1, !j)*r(!j)
		next
		Phi(!k, !k) = (r(!k) - sum_top)/(1 - sum_bottom)
	endif
	for !j = 1 to !k-1
		Phi(!k, !j) = Phi(!k-1, !j) - Phi(!k, !k)*Phi(!k-1, !k-!j)
		Phi(!j, !k) = Phi(!k, !j)
	next
next

series PAC_series
for !k = 1 to @obs(data)
	PAC_series(!k) = Phi(!k, !k)
next

delete weighted_sum
delete sum_top
delete sum_bottom

