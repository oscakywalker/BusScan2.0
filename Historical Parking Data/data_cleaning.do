pwd //print current working directory
cd "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan2.23"

import delimited "2020_Melbourne.csv", clear
save data1.dta, replace

gen double arriv_time = clock(arrivaltime, "MDYhms")
gen double depart_time = clock(departuretime, "MDYhms")

format arriv_time %tc
format depart_time %tc

sort deviceid arriv_time

bysort deviceid: gen count = _N
keep if count > 1000

drop count

gen State = 1

save data1.dta, replace

use data1.dta, clear
gen is_last = 0
bysort deviceid (arriv_time): replace is_last = 1 if _n == _N

// 插入新行（排除最后一行）
gen id = _n // 创建唯一标识符
expand 2 if is_last == 0 // 每行扩展为两行（但不扩展最后一行）

sort deviceid id

gen is_new = mod(_n, 2) == 0 

replace State = 0 if is_new // 新插入的行的 State 设置为 0
replace arriv_time = . if is_new // 新插入的行的 arriv_time 设置为空
replace depart_time = . if is_new // 新插入的行的 depart_time 设置为空
replace lastingtime = . if is_new

replace lastingtime = arriv_time[_n+1] - depart_time[_n-1] if is_new
replace lastingtime = lastingtime / 1000 if is_new

drop arriv_time depart_time is_last id is_new arrivaltime departuretime durationminutes
drop if lastingtime < 0
// 检测 lastingtime == 0 的行并加 1
replace lastingtime = lastingtime + 1 if lastingtime == 0
export delimited "/Users/fujunhan/Desktop/RA/my_research_related_materials/plan2.23/2020Melbourne_clean.csv", replace
