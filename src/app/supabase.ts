import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://zrygncayibskiktbxndz.supabase.co'; // Your Project URL
const supabaseAnonKey = 'sb_publishable_uqcElFRPLPZyoEsdWa-s1w_YoZVjgeZ'; 

export const supabase = createClient(supabaseUrl, supabaseAnonKey);