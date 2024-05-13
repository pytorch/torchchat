#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <dlfcn.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>


#define COUNT_OF(x) (sizeof(x) / sizeof(0[x]))


struct egl_enum_item {
	EGLint id;
	const char* name;
};

struct egl_enum_item egl_enum_boolean[] = {
	{
		.id = EGL_TRUE,
		.name = "EGL_TRUE",
	},
	{
		.id = EGL_FALSE,
		.name = "EGL_FALSE",
	},
};

struct egl_enum_item egl_enum_caveat[] = {
	{
		.id = EGL_NONE,
		.name = "EGL_NONE",
	},
	{
		.id = EGL_SLOW_CONFIG,
		.name = "EGL_SLOW_CONFIG",
	},
	{
		.id = EGL_NON_CONFORMANT_CONFIG,
		.name = "EGL_NON_CONFORMANT_CONFIG",
	},
};

struct egl_enum_item egl_enum_transparency[] = {
	{
		.id = EGL_NONE,
		.name = "EGL_NONE",
	},
	{
		.id = EGL_TRANSPARENT_RGB,
		.name = "EGL_TRANSPARENT_RGB",
	},
};

struct egl_enum_item egl_enum_color_buffer[] = {
	{
		.id = EGL_RGB_BUFFER,
		.name = "EGL_RGB_BUFFER",
	},
	{
		.id = EGL_LUMINANCE_BUFFER,
		.name = "EGL_LUMINANCE_BUFFER",
	},
};

#ifndef EGL_OPENGL_ES3_BIT
	#define EGL_OPENGL_ES3_BIT 0x40
#endif

struct egl_enum_item egl_enum_conformant[] = {
	{
		.id = EGL_OPENGL_BIT,
		.name = "EGL_OPENGL_BIT",
	},
	{
		.id = EGL_OPENGL_ES_BIT,
		.name = "EGL_OPENGL_ES_BIT",
	},
	{
		.id = EGL_OPENGL_ES2_BIT,
		.name = "EGL_OPENGL_ES2_BIT",
	},
	{
		.id = EGL_OPENGL_ES3_BIT,
		.name = "EGL_OPENGL_ES3_BIT",
	},
	{
		.id = EGL_OPENVG_BIT,
		.name = "EGL_OPENVG_BIT",
	},
};

struct egl_enum_item egl_enum_surface_type[] = {
	{
		.id = EGL_PBUFFER_BIT,
		.name = "EGL_PBUFFER_BIT",
	},
	{
		.id = EGL_PIXMAP_BIT,
		.name = "EGL_PIXMAP_BIT",
	},
	{
		.id = EGL_WINDOW_BIT,
		.name = "EGL_WINDOW_BIT",
	},
	{
		.id = EGL_VG_COLORSPACE_LINEAR_BIT,
		.name = "EGL_VG_COLORSPACE_LINEAR_BIT",
	},
	{
		.id = EGL_VG_ALPHA_FORMAT_PRE_BIT,
		.name = "EGL_VG_ALPHA_FORMAT_PRE_BIT",
	},
	{
		.id = EGL_MULTISAMPLE_RESOLVE_BOX_BIT,
		.name = "EGL_MULTISAMPLE_RESOLVE_BOX_BIT",
	},
	{
		.id = EGL_SWAP_BEHAVIOR_PRESERVED_BIT,
		.name = "EGL_SWAP_BEHAVIOR_PRESERVED_BIT",
	},
};

struct egl_enum_item egl_enum_renderable_type[] = {
	{
		.id = EGL_OPENGL_ES_BIT,
		.name = "EGL_OPENGL_ES_BIT",
	},
	{
		.id = EGL_OPENVG_BIT,
		.name = "EGL_OPENVG_BIT",
	},
	{
		.id = EGL_OPENGL_ES2_BIT,
		.name = "EGL_OPENGL_ES2_BIT",
	},
	{
		.id = EGL_OPENGL_BIT,
		.name = "EGL_OPENGL_BIT",
	},
	{
		.id = EGL_OPENGL_ES3_BIT,
		.name = "EGL_OPENGL_ES3_BIT",
	},
};

struct egl_config_attribute {
	EGLint id;
	const char* name;
	int32_t cardinality;
	const struct egl_enum_item* values;
};

struct egl_config_attribute egl_config_attributes[] = {
	{
		.id = EGL_CONFIG_ID,
		.name = "EGL_CONFIG_ID",
	},
	{
		.id = EGL_CONFIG_CAVEAT,
		.name = "EGL_CONFIG_CAVEAT",
		.cardinality = COUNT_OF(egl_enum_caveat),
		.values = egl_enum_caveat,
	},
	{
		.id = EGL_LUMINANCE_SIZE,
		.name = "EGL_LUMINANCE_SIZE",
	},
	{
		.id = EGL_RED_SIZE,
		.name = "EGL_RED_SIZE",
	},
	{
		.id = EGL_GREEN_SIZE,
		.name = "EGL_GREEN_SIZE",
	},
	{
		.id = EGL_BLUE_SIZE,
		.name = "EGL_BLUE_SIZE",
	},
	{
		.id = EGL_ALPHA_SIZE,
		.name = "EGL_ALPHA_SIZE",
	},
	{
		.id = EGL_DEPTH_SIZE,
		.name = "EGL_DEPTH_SIZE",
	},
	{
		.id = EGL_STENCIL_SIZE,
		.name = "EGL_STENCIL_SIZE",
	},
	{
		.id = EGL_ALPHA_MASK_SIZE,
		.name = "EGL_ALPHA_MASK_SIZE",
	},
	{
		.id = EGL_BIND_TO_TEXTURE_RGB,
		.name = "EGL_BIND_TO_TEXTURE_RGB",
		.cardinality = COUNT_OF(egl_enum_boolean),
		.values = egl_enum_boolean,
	},
	{
		.id = EGL_BIND_TO_TEXTURE_RGBA,
		.name = "EGL_BIND_TO_TEXTURE_RGBA",
		.cardinality = COUNT_OF(egl_enum_boolean),
		.values = egl_enum_boolean,
	},
	{
		.id = EGL_MAX_PBUFFER_WIDTH,
		.name = "EGL_MAX_PBUFFER_WIDTH",
	},
	{
		.id = EGL_MAX_PBUFFER_HEIGHT,
		.name = "EGL_MAX_PBUFFER_HEIGHT",
	},
	{
		.id = EGL_MAX_PBUFFER_PIXELS,
		.name = "EGL_MAX_PBUFFER_PIXELS",
	},
	{
		.id = EGL_TRANSPARENT_RED_VALUE,
		.name = "EGL_TRANSPARENT_RED_VALUE",
	},
	{
		.id = EGL_TRANSPARENT_GREEN_VALUE,
		.name = "EGL_TRANSPARENT_GREEN_VALUE",
	},
	{
		.id = EGL_TRANSPARENT_BLUE_VALUE,
		.name = "EGL_TRANSPARENT_BLUE_VALUE",
	},
	{
		.id = EGL_SAMPLE_BUFFERS,
		.name = "EGL_SAMPLE_BUFFERS",
	},
	{
		.id = EGL_SAMPLES,
		.name = "EGL_SAMPLES",
	},
	{
		.id = EGL_LEVEL,
		.name = "EGL_LEVEL",
	},
	{
		.id = EGL_MAX_SWAP_INTERVAL,
		.name = "EGL_MAX_SWAP_INTERVAL",
	},
	{
		.id = EGL_MIN_SWAP_INTERVAL,
		.name = "EGL_MIN_SWAP_INTERVAL",
	},
	{
		.id = EGL_SURFACE_TYPE,
		.name = "EGL_SURFACE_TYPE",
		.cardinality = -(int32_t) COUNT_OF(egl_enum_surface_type),
		.values = egl_enum_surface_type,
	},
	{
		.id = EGL_RENDERABLE_TYPE,
		.name = "EGL_RENDERABLE_TYPE",
		.cardinality = -(int32_t) COUNT_OF(egl_enum_renderable_type),
		.values = egl_enum_renderable_type,
	},
	{
		.id = EGL_CONFORMANT,
		.name = "EGL_CONFORMANT",
		.cardinality = -(int32_t) COUNT_OF(egl_enum_conformant),
		.values = egl_enum_conformant,
	},
	{
		.id = EGL_TRANSPARENT_TYPE,
		.name = "EGL_TRANSPARENT_TYPE",
		.cardinality = COUNT_OF(egl_enum_transparency),
		.values = egl_enum_transparency,
	},
	{
		.id = EGL_COLOR_BUFFER_TYPE,
		.name = "EGL_COLOR_BUFFER_TYPE",
		.cardinality = COUNT_OF(egl_enum_color_buffer),
		.values = egl_enum_color_buffer,
	},
};

void report_gles_attributes(void) {
	void* libEGL = NULL;
	EGLConfig* configs = NULL;
	EGLDisplay display = EGL_NO_DISPLAY;
	EGLSurface surface = EGL_NO_SURFACE;
	EGLContext context = EGL_NO_CONTEXT;
	EGLBoolean egl_init_status = EGL_FALSE;
	EGLBoolean egl_make_current_status = EGL_FALSE;
	EGLBoolean egl_status;

	libEGL = dlopen("libEGL.so", RTLD_LAZY | RTLD_LOCAL);

	display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (display == EGL_NO_DISPLAY) {
		fprintf(stderr, "failed to get default EGL display\n");
		goto cleanup;
	}

	EGLint egl_major = 0, egl_minor = 0;
	egl_init_status = eglInitialize(display, &egl_major, &egl_minor);
	if (egl_init_status != EGL_TRUE) {
		fprintf(stderr, "failed to initialize EGL display connection\n");
		goto cleanup;
	}
	printf("initialized display connection with EGL %d.%d\n", (int) egl_major, (int) egl_minor);

	EGLint configs_count = 0;
	egl_status = eglGetConfigs(display, NULL, 0, &configs_count);
	if (egl_status != EGL_TRUE) {
		fprintf(stderr, "failed to get the number of EGL frame buffer configurations\n");
		goto cleanup;
	}

	configs = (EGLConfig*) malloc(configs_count * sizeof(EGLConfig));
	if (configs == NULL) {
		fprintf(stderr, "failed to allocate %zu bytes for %d frame buffer configurations\n",
			configs_count * sizeof(EGLConfig), configs_count);
		goto cleanup;
	}

	egl_status = eglGetConfigs(display, configs, configs_count, &configs_count);
	if (egl_status != EGL_TRUE || configs_count == 0) {
		fprintf(stderr, "failed to get EGL frame buffer configurations\n");
		goto cleanup;
	}

	printf("EGL framebuffer configurations:\n");
	for (EGLint i = 0; i < configs_count; i++) {
		printf("\tConfiguration #%d:\n", (int) i);
		for (size_t n = 0; n < COUNT_OF(egl_config_attributes); n++) {
			EGLint value = 0;
			egl_status = eglGetConfigAttrib(display, configs[i], egl_config_attributes[n].id, &value);
			if (egl_config_attributes[n].cardinality == 0) {
				printf("\t\t%s: %d\n", egl_config_attributes[n].name, (int) value);
			} else if (egl_config_attributes[n].cardinality > 0) {
				/* Enumeration */
				bool known_value = false;
				for (size_t k = 0; k < (size_t) egl_config_attributes[n].cardinality; k++) {
					if (egl_config_attributes[n].values[k].id == value) {
						printf("\t\t%s: %s\n", egl_config_attributes[n].name, egl_config_attributes[n].values[k].name);
						known_value = true;
						break;
					}
				}
				if (!known_value) {
					printf("\t\t%s: unknown (%d)\n", egl_config_attributes[n].name, value);
				}
			} else {
				/* Bitfield */
				printf("\t\t%s: ", egl_config_attributes[n].name);
				if (value == 0) {
					printf("none\n");
				} else {
					for (size_t k = 0; k < (size_t) -egl_config_attributes[n].cardinality; k++) {
						if (egl_config_attributes[n].values[k].id & value) {
							value &= ~egl_config_attributes[n].values[k].id;
							if (value != 0) {
								printf("%s | ", egl_config_attributes[n].values[k].name);
							} else {
								printf("%s\n", egl_config_attributes[n].values[k].name);
							}
						}
					}
					if (value != 0) {
						printf("0x%08X\n", (int) value);
					}
				}
			}
		}
	}

	EGLint const config_attributes[] = {
		EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_CONFORMANT, EGL_OPENGL_ES2_BIT,
		EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
		EGL_NONE,
	};
	EGLConfig config = NULL;
	EGLint config_count = 0;
	egl_status = eglChooseConfig(display, config_attributes, &config, 1, &config_count);
	if (egl_status != EGL_TRUE || config_count == 0 || config == NULL) {
		fprintf(stderr, "failed to find EGL frame buffer configuration that match required attributes\n");
		goto cleanup;
	}

	EGLint const surface_attributes[] = {
		EGL_HEIGHT, 1,
		EGL_WIDTH, 1,
		EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA,
		EGL_TEXTURE_TARGET, EGL_TEXTURE_2D,
		EGL_NONE,
	};
	surface = eglCreatePbufferSurface(display, config, surface_attributes);
	if (surface == EGL_NO_SURFACE) {
		fprintf(stderr, "failed to create PBuffer surface\n");
		goto cleanup;
	}

	EGLint const context_attributes[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE,
	};
	context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
	if (context == EGL_NO_CONTEXT) {
		fprintf(stderr, "failed to create OpenGL ES context\n");
		goto cleanup;
	}

	egl_make_current_status = eglMakeCurrent(display, surface, surface, context);
	if (egl_make_current_status != EGL_TRUE) {
		fprintf(stderr, "failed to attach OpenGL ES rendering context\n");
		goto cleanup;
	}

	printf("OpenGL ES Attributes:\n");
	printf("\t%s: \"%s\"\n", "GL_VENDOR", glGetString(GL_VENDOR));
	printf("\t%s: \"%s\"\n", "GL_RENDERER", glGetString(GL_RENDERER));
	printf("\t%s: \"%s\"\n", "GL_VERSION", glGetString(GL_VERSION));
	printf("\t%s: \"%s\"\n", "GL_SHADING_LANGUAGE_VERSION", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("\t%s: \"%s\"\n", "GL_EXTENSIONS", glGetString(GL_EXTENSIONS));

cleanup:
	if (egl_make_current_status == EGL_TRUE) {
		eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	}
	if (context != EGL_NO_CONTEXT) {
		eglDestroyContext(display, context);
	}
	if (surface != EGL_NO_SURFACE) {
		eglDestroySurface(display, surface);
	}
	if (egl_init_status == EGL_TRUE) {
		eglTerminate(display);
	}
	free(configs);

	if (libEGL != NULL) {
		dlclose(libEGL);
	}
}

int main(int argc, char** argv) {
	report_gles_attributes();
	return 0;
}
