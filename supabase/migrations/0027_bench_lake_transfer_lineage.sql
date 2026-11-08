-- Replace the authoritative lake lineage in one operator action.
--
-- The two-argument bench_lake_transfer moves the lease between machines.  A
-- replacement lake built from a copy also has a NEW lake_id; without this
-- overload an operator had to hand-edit the lease row while transferring it.

CREATE FUNCTION public.bench_lake_transfer(
    p_old_machine text,
    p_new_machine text,
    p_old_lake uuid,
    p_new_lake uuid
) RETURNS text
    LANGUAGE plpgsql
    SET search_path TO 'public'
    AS $$
BEGIN
    IF coalesce(p_new_machine, '') = '' OR p_old_lake IS NULL
       OR p_new_lake IS NULL OR p_old_lake = p_new_lake THEN
        RAISE EXCEPTION
            'bench_lake_transfer: p_new_machine and two distinct lake ids are required';
    END IF;

    UPDATE public.bench_lake_lease
       SET lake_id = p_new_lake,
           machine = p_new_machine,
           claimed_at = now()
     WHERE id AND machine = p_old_machine AND lake_id = p_old_lake;

    IF FOUND THEN
        RETURN 'moved';
    END IF;
    RETURN 'refused';
END
$$;

REVOKE ALL ON FUNCTION public.bench_lake_transfer(text, text, uuid, uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.bench_lake_transfer(text, text, uuid, uuid)
    FROM service_role, authenticated, anon;
