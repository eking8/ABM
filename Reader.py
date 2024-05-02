from line_profiler import LineProfiler

def read_profile_data(filename):
    # Create a LineProfiler object
    lp = LineProfiler()
    
    # Load the profile data from the file
    lp.show_results(filename)
    
    # Print the formatted profile data
    lp.print_stats()

# Specify the .lprof file
profile_file = 'Run4.py.lprof'
read_profile_data(profile_file)