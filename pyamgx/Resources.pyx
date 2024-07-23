cdef class Resources:
    """
    Resources: Class for creating and freeing AMGX Resources objects.
    """
    cdef AMGX_resources_handle rsrc

    def create_simple(self, Config cfg):
        """
        rsc.create_simple(cfg)

        Create the underlying AMGX Resources object in a
        single-threaded application.

        Parameters
        ----------
        cfg : Config

        Returns
        -------
        self : Resources
        """
        check_error(AMGX_resources_create_simple(&self.rsrc, cfg.cfg))
        return self
    
    def create_parallel(self, Config cfg, device_num):
        """
        AFW 2024
        AMGX_resources_create

        Create the underlying AMGX Resources object in a
        single-threaded application on _multiple GPU_

        Parameters
        ----------
        cfg : Config

        Returns
        -------
        self : Resources
        """
        # Use NULL comm
        cdef uintptr_t devices = ptr_from_array_interface(
            np.array([device_num], dtype=np.int32), "int32"
        )
        
        check_error(AMGX_resources_create(&self.rsrc, cfg.cfg, NULL,  1, <const int*> devices))
        return self

    def destroy(self):
        """
        rsc.destroy()

        Destroy the underlying AMGX Resources object.
        """
        check_error(AMGX_resources_destroy(self.rsrc))
