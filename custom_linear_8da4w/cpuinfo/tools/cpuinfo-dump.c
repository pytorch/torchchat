#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>


#define BUFFER_SIZE 4096
char buffer[BUFFER_SIZE];

#define CPUINFO_PATH "/proc/cpuinfo"

int main(int argc, char** argv) {
	int file = open(CPUINFO_PATH, O_RDONLY);
	if (file == -1) {
		fprintf(stderr, "Error: failed to open %s: %s\n", CPUINFO_PATH, strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* Only used for error reporting */
	size_t position = 0;
	char* data_start = buffer;
	ssize_t bytes_read;
	do {
		bytes_read = read(file, buffer, BUFFER_SIZE);
		if (bytes_read < 0) {
			fprintf(stderr, "Error: failed to read file %s at position %zu: %s\n",
				CPUINFO_PATH, position, strerror(errno));
			exit(EXIT_FAILURE);
		}

		position += (size_t) bytes_read;
		if (bytes_read > 0) {
			fwrite(buffer, 1, (size_t) bytes_read, stdout);
		}
	} while (bytes_read != 0);

	if (close(file) != 0) {
		fprintf(stderr, "Error: failed to close %s: %s\n", CPUINFO_PATH, strerror(errno));
		exit(EXIT_FAILURE);
	}
	return EXIT_SUCCESS;
}
