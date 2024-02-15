import os, sys, glob, tqdm

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--sample",                 dest="sample",                 default="",                           help="Sample name on CMSDAS")
    parser.add_option("--txt",                    dest="txt",                    default="",                           help="Name of the txt")
    (options, args) = parser.parse_args()
    print(options)

    sample_name = options.sample
    txt_name = os.getcwd() + "/inputFiles/"  + "__".join(sample_name.split('/')[1:]) + '.txt'
    print(f" ### INFO: Saving samples to {txt_name}")

    avaliable_sample_list = []

    # Get all available samples
    cmd = 'dasgoclient --query=="file dataset={}"'.format(sample_name)
    sample_list = os.popen(cmd).read().split('\n')[:-1]

    for sample in tqdm.tqdm(sample_list):
        file_name = sample.split('\n')[0]
        cmd = 'dasgoclient --query=="site file={}"'.format(file_name)
        sites = os.popen(cmd).read().split('\n')[:-1]
        tape = [int("Tape" not in site) for site in sites]
        sum_site = sum(int(x) for x in tape)
        if sum_site > 0:
            avaliable_sample_list.append(sample)

    file = open (txt_name, 'w')
    for line in avaliable_sample_list:
        file.write (line)
    file.close ()
