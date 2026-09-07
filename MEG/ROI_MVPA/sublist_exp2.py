# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:02:19 2022

@author: Ling_BLCU
"""

# -*- coding: utf-8 -*-
"""
===========
Subject list file
===========

Configurate the parameters of the subject.

"""

# =============================================================================
# subject_info
# =============================================================================

# subject_list
sub_list=dict()
#AT_cmb_stf_full
sub_a_list = ['SA101', 'SA104', 'SA108', 'SA111','SA112',    #,
              'SA114','SA116', 'SA118', 'SA121', 'SA123',
              'SA124','SA126','SA127','SA131','SA132',     #,
              'SA136','SA138','SA139','SA140','SA144',
              'SA145','SA146','SA148','SA150','SA151',
              'SA154','SA160','SA166','SA170','SA173',
              'SA174','SA176' ]                                        
sub_b_list = ['SB006','SB008','SB009','SB011', 'SB015',
              'SB016','SB019','SB023','SB024', 'SB030',                 #,
              'SB031','SB036', 'SB039','SB040','SB041',
              'SB042','SB044', 'SB050','SB051','SB061',
              'SB063','SB069', 'SB071','SB074','SB081']
sub_list['AT_cmb_stf_full'] = sub_a_list + sub_b_list

#AT_cmb_stf_full_loc
sub_a_list1 = ['SA101', 'SA104', 'SA108', 'SA111','SA112',    #,
              'SA114','SA116', 'SA118', 'SA121', 'SA123',
              'SA124','SA126','SA127','SA131','SA132',     #,
              'SA136','SA138','SA139','SA140','SA144',
              'SA145','SA146','SA148','SA150','SA151',
              'SA154','SA160','SA166','SA170','SA173',
              'SA174','SA176' ]                                        
sub_b_list1= ['SB006','SB008','SB009','SB011', 'SB015',
              'SB016','SB019','SB023','SB024', 'SB028','SB030',                 #,
              'SB031','SB036', 'SB039','SB040','SB041',
              'SB042','SB044', 'SB050','SB051','SB061',
              'SB063','SB069', 'SB071','SB073','SB074','SB081']
sub_list['AT_cmb_stf_full_loc'] = sub_a_list1 + sub_b_list1


#dAT_full
sub_a_list2 = ['SA114','SA121','SA123', 'SA124','SA126', 
               'SA127','SA131','SA136','SA138','SA144',
               'SA151','SA154','SA160','SA166','SA173',
               'SA174']                                        
sub_b_list2= ['SB006','SB009','SB011','SB013', 'SB015',
              'SB016','SB023','SB024','SB028','SB031',
              'SB044','SB049','SB050','SB061','SB063',
              'SB073','SB074','SB084','SB999']
sub_list['dAT_full'] = sub_a_list2 + sub_b_list2
sub_list['dAT_subPFC_full'] = sub_a_list2 + sub_b_list2



#dAT_loc_full
sub_a_list3 = ['SA104','SA111','SA112', 'SA114','SA121', 
               'SA123','SA124','SA126','SA127','SA136',
               'SA144','SA148','SA151','SA154','SA173']                                        
sub_b_list3= ['SB006','SB009','SB013', 'SB015','SB023',
              'SB031','SB049','SB061','SB063','SB073',
              'SB074','SB084','SB999']
sub_list['dAT_loc_full'] = sub_a_list3 + sub_b_list3
sub_list['dAT_subPFC_loc_full'] = sub_a_list3 + sub_b_list3
sub_list['dAT_subP2F_loc_full'] = sub_a_list3 + sub_b_list3

#ET_WCD
sub_a_list33 = ['SA104','SA111','SA112', 'SA114','SA121', 
               'SA123','SA124','SA126','SA127','SA136',
               'SA144','SA148','SA154','SA173'] #SA151 had bad ET                                       
sub_b_list33= ['SB006','SB009','SB013', 'SB015','SB023',
              'SB031','SB049','SB061','SB063','SB073',
              'SB074','SB084','SB999']
sub_list['ET_WCD'] = sub_a_list33 + sub_b_list33


#GAT_dAT_full
sub_a_list4 = ['SA114','SA121','SA123', 'SA124','SA126', 
               'SA127','SA131','SA136','SA138','SA144',
               'SA151','SA154','SA160','SA166','SA173',
               'SA174']                                        
sub_b_list4= ['SB006','SB009','SB011','SB013', 'SB015',
              'SB016','SB023','SB024','SB028','SB031',
              'SB044','SB049','SB050','SB061','SB063',
              'SB073','SB074','SB084','SB999']
sub_list['GAT_PFC_dAT_full'] = sub_a_list4 + sub_b_list4



#GAT_dAT_loc_full
sub_a_list5 = ['SA104','SA111','SA112', 'SA114','SA121', 
               'SA123','SA124','SA126','SA127','SA136',
               'SA144','SA148','SA151','SA154','SA173']                                        
sub_b_list5= ['SB006','SB009','SB013', 'SB015','SB023',
              'SB031','SB049','SB061','SB063','SB073',
              'SB074','SB084','SB999']
sub_list['GAT_PFC_dAT_loc_full'] = sub_a_list5 + sub_b_list5


#seen_full
sub_a_list6 = ['SA114','SA121','SA123', 'SA124','SA126', 
               'SA127','SA131','SA136','SA138','SA144',
               'SA151','SA154','SA160','SA166','SA173',
               'SA174']                                        
sub_b_list6= ['SB006','SB009','SB011','SB013', 'SB015',
              'SB016','SB023','SB024','SB028','SB031',
              'SB044','SB049','SB050','SB061','SB063',
              'SB073','SB074','SB084','SB999']
sub_list['seen_full'] = sub_a_list6 + sub_b_list6
sub_list['seen_subPFC_full'] = sub_a_list6 + sub_b_list6

#CCT_PFC_full(dAT_seen, AT_nogo_seen,AT_go_seen)
sub_a_list7 = ['SA104','SA111','SA116',#'SA114','SA118',
               'SA121','SA123','SA124','SA126','SA127',
               'SA131','SA132','SA136','SA138','SA139',
               'SA140','SA144','SA145','SA146','SA150',
               'SA151','SA154','SA160','SA166','SA173',
               'SA174','SA176']                                        
sub_b_list7= ['SB006','SB008','SB009','SB011','SB015',
              'SB016','SB023','SB024','SB031','SB036',
              'SB039','SB040','SB042','SB044','SB050',
              'SB061','SB063','SB069','SB071','SB074',
              'SB999']
sub_list['CCD_PFC_full'] = sub_a_list7 + sub_b_list7
sub_list['CCD_subPFC_full'] = sub_a_list7 + sub_b_list7
sub_list['CTCCD_PFC_full'] = sub_a_list7 + sub_b_list7
sub_list['WCD_full'] = sub_a_list7 + sub_b_list7
sub_list['WCD_subPFC_full'] = sub_a_list7 + sub_b_list7

#CCT_IIT_full(dAT_seen AT_nogo_seen)
sub_a_list8 = ['SA104','SA111','SA114','SA116','SA118',
               'SA121','SA123','SA124','SA126','SA127',
               'SA132','SA136','SA138','SA139','SA140',
               'SA144','SA145','SA146','SA150','SA151',
               'SA154','SA160','SA170','SA173','SA174',
               'SA176']                                        
sub_b_list8= ['SB006','SB008','SB009','SB011','SB015',
              'SB023','SB024','SB031','SB036','SB039',
              'SB040','SB042','SB044','SB049','SB051',
              'SB061','SB063','SB069','SB071','SB073',
              'SB074','SB999']
sub_list['CCD_IIT_full'] = sub_a_list8 + sub_b_list8

