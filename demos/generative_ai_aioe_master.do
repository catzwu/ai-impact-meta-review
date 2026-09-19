cd "C:\Users\rcs10\Dropbox\AESG\AIOE"

ssc install binscatter 
//ssc install egen

*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************
*FIGURE 1
*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************
*******************************************************************************************************************************************************************************

use generative_ai_aioe, clear

merge 1:1 occ_code using occ_salary_data_2021
drop if _merge==2
drop _merge

merge 1:1 occ_code using occ_required_education
drop _merge

merge 1:1 occ_code using occ_creative_weight
drop _merge


keep occ_code occ_req_education median_salary_2021 avg_creative_weight lm_aioe ig_aioe

*Rename to reshape -- extra underscore for LM so it shows up first
rename ig_aioe aioe_ig
rename lm_aioe aioe__lm

reshape long aioe, i(occ_code) j(application, string)

binscatter aioe median_salary_2021, savedata(mysalarybins) by(application) line(none) xtitle("Median Salary 2021" "(Thousands of Dollars)") ytitle("Exposure to Generative AI") legend(lab(1 "Language Modeling")lab(2 "Image Generation") ) msymbol(O D) mcolor(dknavy blue*.5) yline(0, lcolor(black))
preserve
import delimited "mysalarybins.csv", clear
restore

binscatter aioe occ_req_education, savedata(myeducationbins) by(application) line(none) xtitle("Occupation Required Education Level") ytitle("Exposure to Generative AI") msymbol(O D) legend(lab(1 "Language Modeling AIOE")lab(2 "Image Generation AIOE") )  mcolor(dknavy blue*.5) yline(0, lcolor(black)) xlabel(0(2)12)
preserve
import delimited "myeducationbins.csv", clear
restore

binscatter aioe avg_creative_weight, savedata(mycreativebins) by(application) line(none) xtitle("Relative Weight of Creative Abilities") ytitle("Exposure to Generative AI") msymbol(O D) legend(lab(1 "Language Modeling AIOE")lab(2 "Image Generation AIOE") )  mcolor(dknavy blue*.5) yline(0, lcolor(black)) xlabel(0 "0%" .02 "2%" .04 "4%" .06 "6%" .08 "8%" .1 "10%")
preserve
import delimited "mycreativebins.csv", clear
restore


binscatter aioe median_salary_2021, by(application) line(none) xtitle("Median Salary 2021" "(Thousands of Dollars)") ytitle("Exposure to Generative AI") legend(lab(1 "Language Modeling")lab(2 "Image Generation") ) msymbol(O D) mcolor(dknavy blue*.5) yline(0, lcolor(black))

binscatter aioe occ_req_education, by(application) line(none) xtitle("Occupation Required Education Level") ytitle("Exposure to Generative AI") msymbol(O D) legend(lab(1 "Language Modeling AIOE")lab(2 "Image Generation AIOE") )  mcolor(dknavy blue*.5) yline(0, lcolor(black)) xlabel(0(2)12)

binscatter aioe avg_creative_weight, by(application) line(none) xtitle("Relative Weight of Creative Abilities") ytitle("Exposure to Generative AI") msymbol(O D) legend(lab(1 "Language Modeling AIOE")lab(2 "Image Generation AIOE") )  mcolor(dknavy blue*.5) yline(0, lcolor(black)) xlabel(0 "0%" .02 "2%" .04 "4%" .06 "6%" .08 "8%" .1 "10%")

