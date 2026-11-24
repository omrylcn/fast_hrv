use std::io::Read;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = &args[1];
    let fs: u32 = args[2].parse().unwrap();
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let v: Vec<f64> = s.split_whitespace().map(|x| x.parse().unwrap()).collect();
    match mode.as_str() {
        "clean" => for x in fast_hrv::ecg_clean(&v, fs) { println!("{x}") },
        "peaks" => for x in fast_hrv::find_peaks_default(&v, fs) { println!("{x}") },
        "fix" => {
            let p: Vec<usize> = v.iter().map(|&x| x as usize).collect();
            let (f, a) = fast_hrv::fix_peaks_default(&p, fs);
            println!("# ectopic {:?} missed {:?} extra {:?} longshort {:?}", a.ectopic, a.missed, a.extra, a.longshort);
            for x in f { println!("{x}") }
        }
        "quality" => {
            let r = fast_hrv::process_ecg(&v, fs, false);
            let (_, q) = fast_hrv::compute_quality(&r.ecg_cleaned, &r.rpeaks, fs);
            for x in q { println!("{x}") }
        }
        _ => panic!(),
    }
}
