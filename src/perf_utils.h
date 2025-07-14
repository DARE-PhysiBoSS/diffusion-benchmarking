#include <papi.h>
#include <source_location>
#include <sstream>

class perf_counter
{
	std::string region_;
	const std::source_location location_;

	static bool enabled_;

	void check_error(int retval)
	{
		if (retval != PAPI_OK)
		{
			std::ostringstream ss;
			ss << location_.file_name() << '(' << location_.line() << ':' << location_.column() << ')';
			PAPI_perror(ss.str().c_str());
			std::exit(retval);
		}
	}

public:
	static void enable() { enabled_ = true; }

	perf_counter(std::string region_name, const std::source_location location = std::source_location::current())
		: region_(region_name), location_(location)
	{
		if (!enabled_)
			return;

		check_error(PAPI_hl_region_begin(region_.c_str()));
	}

	~perf_counter()
	{
		if (!enabled_)
			return;

		check_error(PAPI_hl_region_end(region_.c_str()));
	}
};
